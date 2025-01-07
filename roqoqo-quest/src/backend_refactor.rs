// Copyright © 2021 HQS Quantum Simulations GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use crate::interface::{
    call_operation_with_device, execute_pragma_repeated_measurement, initialize_registers,
};

use roqoqo::backends::EvaluatingBackend;
use roqoqo::backends::RegisterResult;
use roqoqo::operations::*;
use roqoqo::registers::{BitRegister, ComplexRegister, FloatRegister};
use roqoqo::Circuit;
use roqoqo::RoqoqoBackendError;

use crate::Qureg;
use std::collections::{HashMap, HashSet};
use std::panic::resume_unwind;

#[cfg(feature = "async")]
use async_trait::async_trait;
#[cfg(feature = "async")]
use roqoqo::backends::AsyncEvaluatingBackend;

#[cfg(feature = "parallelization")]
use rayon::prelude::*;
#[cfg(feature = "parallelization")]
use roqoqo::measurements::Measure;
#[cfg(feature = "parallelization")]
use roqoqo::registers::Registers;

const REPEATED_MEAS_ERROR: &str =
    "Only one repeated measurement allowed in the circuit. Make sure \
     that the submitted circuit contains only one \
     PragmaRepeatedMeasurement or one \
     PragmaSetNumberOfMeasurements.";

/// QuEST backend
///
/// Provides functions to run circuits and measurements with the QuEST quantum simulator.
/// If different instances of the backend are running in parallel, the results won't be deterministic,
/// even with a random_seed set.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Backend {
    /// Number of qubits supported by the backend
    pub number_qubits: usize,
    /// Random seed
    pub random_seed: Option<Vec<u64>>,
}

impl Backend {
    /// Creates a new QuEST backend.
    ///
    /// # Arguments
    ///
    /// * `number_qubits` - The number of qubits supported by the backend
    pub fn new(number_qubits: usize, random_seed: Option<Vec<u64>>) -> Self {
        Self {
            number_qubits,
            random_seed,
        }
    }

    /// Sets the random random seed for the backend.
    /// If different instances of the backend are running in parallel, the results won't be deterministic,
    /// even with a random_seed set.
    ///
    /// # Arguments
    ///
    /// * `random_seed` - The random seed to use for the backend
    pub fn set_random_seed(&mut self, random_seed: Vec<u64>) {
        self.random_seed = Some(random_seed);
    }

    /// Gets the current random seed set for the backend.
    ///
    /// # Returns
    ///
    /// * `Option<Vec<u64>>` - The current random seed
    pub fn get_random_seed(&self) -> Option<Vec<u64>> {
        self.random_seed.clone()
    }
}

impl EvaluatingBackend for Backend {
    fn run_circuit_iterator<'a>(
        &self,
        circuit: impl Iterator<Item = &'a Operation>,
    ) -> RegisterResult {
        // TODO discuss this. I find this approach much cleaner because from here on we only deal
        // with roqoqo Circuit objects. But we have the overhead of copying the whole circuit.
        let circuit = circuit.into_iter().cloned().collect::<Circuit>();
        self.run_circuit_iterator_with_device(circuit, &mut None)
    }

    #[cfg(feature = "parallelization")]
    fn run_measurement_registers<T>(&self, measurement: &T) -> RegisterResult
    where
        T: Measure,
    {
        let mut bit_registers: HashMap<String, BitOutputRegister> = HashMap::new();
        let mut float_registers: HashMap<String, FloatOutputRegister> = HashMap::new();
        let mut complex_registers: HashMap<String, ComplexOutputRegister> = HashMap::new();

        let circuits: Vec<&Circuit> = measurement.circuits().collect();
        let constant_circuit = measurement.constant_circuit();

        let tmp_regs_res: Result<Vec<Registers>, RoqoqoBackendError> = circuits
            .par_iter()
            .map(|circuit| match constant_circuit {
                Some(x) => self.run_circuit_iterator(x.iter().chain(circuit.iter())),
                None => self.run_circuit_iterator(circuit.iter()),
            })
            .collect();
        let tmp_regs = tmp_regs_res?;

        for (tmp_bit_reg, tmp_float_reg, tmp_complex_reg) in tmp_regs.into_iter() {
            for (key, mut val) in tmp_bit_reg.into_iter() {
                if let Some(x) = bit_registers.get_mut(&key) {
                    x.append(&mut val);
                } else {
                    let _ = bit_registers.insert(key, val);
                }
            }
            for (key, mut val) in tmp_float_reg.into_iter() {
                if let Some(x) = float_registers.get_mut(&key) {
                    x.append(&mut val);
                } else {
                    let _ = float_registers.insert(key, val);
                }
            }
            for (key, mut val) in tmp_complex_reg.into_iter() {
                if let Some(x) = complex_registers.get_mut(&key) {
                    x.append(&mut val);
                } else {
                    let _ = complex_registers.insert(key, val);
                }
            }
        }
        Ok((bit_registers, float_registers, complex_registers))
    }
}

impl Backend {
    /// Runs each operation in a quantum circuit on the backend.
    ///
    /// An iterator over operations is passed to the backend and executed. During execution values
    /// are written to and read from classical registers ([crate::registers::BitRegister],
    /// [crate::registers::FloatRegister] and [crate::registers::ComplexRegister]). To produce
    /// sufficient statistics for evaluating expectationg values, circuits have to be run multiple
    /// times. The results of each repetition are concatenated in OutputRegisters
    /// ([crate::registers::BitOutputRegister], [crate::registers::FloatOutputRegister] and
    /// [crate::registers::ComplexOutputRegister]).
    ///
    /// When the optional device parameter is not None availability checks will be performed.
    /// The availability of the operation on a specific device is checked first.
    /// The function returns an error if the operation is not available on the device
    /// even if it can be simulated with the QuEST simulator.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The iterator over operations that is run on the backend (corresponds to a circuit).
    /// * `device` - The optional [roqoqo::devices::Device] that determines the availability of operations
    ///
    /// # Returns
    ///
    /// * `RegisterResult` - The output registers written by the evaluated circuits.
    pub fn run_circuit_iterator_with_device<'a>(
        &self,
        circuit: Circuit,
        device: &mut Option<Box<dyn roqoqo::devices::Device>>,
    ) -> RegisterResult {
        // Initialize output register names from circuit definitions
        let (output_registers, bit_registers_lengths, number_used_qubits) =
            initialize_registers(&circuit)?;
        // unpack
        let (mut bit_registers_output, mut float_registers_output, mut complex_registers_output) =
            output_registers;

        // General circuit validation
        self.validate_circuit(&circuit, number_used_qubits)?;

        // TODO not used at the moment. We would need to adjust all the tests in the other HQS
        // modules if we accounted for global phases.
        // let global_phase = circuit_vec
        //     .iter()
        //     .filter_map(|x| match x {
        //         Operation::PragmaGlobalPhase(x) => Some(x.phase()),
        //         _ => None,
        //     })
        //     .fold(CalculatorFloat::ZERO, |acc, x| acc + x);

        // Switch to density matrix mode if operations are present in the circuit that require
        // density matrix mode
        let is_density_matrix = circuit.iter().any(find_pragma_op);
        // Readout register name for the repeated measurement operation (PragmaRepeatedMeasurement
        // or PragmaSetNumberOfMeasurements), if present
        let mut repeated_measurement_readout: Option<String> = None;
        // how many times to execute repeated measurements
        let mut number_measurements: Option<usize> = None;
        // how many times to simulate the whole circuit from the beginning
        let mut repetitions: usize = 1;
        // Repeated measurement that replaces MeasureQubit operations when needed
        let mut replacement_measurement: Option<PragmaRepeatedMeasurement> = None;
        // TEMP for lack of a better idea
        let mut replace_measurements: Option<usize> = None;

        // simulation_repetitions is the number of times the whole circuit should be rerun for a
        // stochastic simulation. This is the number set with PragmaSimulationRepetitions.
        // number_measurements is the number of shots for repeated measurements, set either with
        // PragmaSetNumberOfMeasurements or with PragmaRepeatedMeasurement.
        handle_repeated_measurements(
            &circuit,
            &mut repeated_measurement_readout,
            &mut number_measurements,
            &mut repetitions,
            &mut replacement_measurement,
            &mut replace_measurements,
            &bit_registers_lengths,
        )?;

        println!("repetitions: {:?}", repetitions);
        println!("replacement_measurement: {:?}", replacement_measurement);
        println!("replace_measurements: {:?}", replace_measurements);
        println!("number_measurements: {:?}", number_measurements);

        let mut qureg = Qureg::new((number_used_qubits) as u32, is_density_matrix);

        if let Some(mut random_seed) = self.random_seed.clone() {
            unsafe {
                quest_sys::seedQuEST(
                    &mut qureg.quest_env,
                    random_seed.as_mut_ptr() as *mut std::os::raw::c_ulong,
                    random_seed.len() as i32,
                );
            };
        }

        // run the actual simulation
        for _ in 0..repetitions {
            qureg.reset();
            let mut bit_registers_internal: HashMap<String, BitRegister> = HashMap::new();
            let mut float_registers_internal: HashMap<String, FloatRegister> = HashMap::new();
            let mut complex_registers_internal: HashMap<String, ComplexRegister> = HashMap::new();
            run_inner_circuit_loop(
                &bit_registers_lengths,
                &circuit,
                (replace_measurements, &replacement_measurement),
                &mut qureg,
                (
                    &mut bit_registers_internal,
                    &mut float_registers_internal,
                    &mut complex_registers_internal,
                ),
                &mut bit_registers_output,
                device,
                number_measurements,
            )?;

            // Append bit result of one circuit execution to output register
            for (name, register) in bit_registers_output.iter_mut() {
                if let Some(tmp_reg) = bit_registers_internal.get(name) {
                    // TODO why is this check here?
                    // if name != &repeated_measurement_readout {
                    register.push(tmp_reg.to_owned())
                    // }
                }
            }
            // Append float result of one circuit execution to output register
            for (name, register) in float_registers_output.iter_mut() {
                if let Some(tmp_reg) = float_registers_internal.get(name) {
                    register.push(tmp_reg.to_owned())
                }
            }
            // Append complex result of one circuit execution to output register
            for (name, register) in complex_registers_output.iter_mut() {
                if let Some(tmp_reg) = complex_registers_internal.get(name) {
                    register.push(tmp_reg.to_owned())
                }
            }
        }
        Ok((
            bit_registers_output,
            float_registers_output,
            complex_registers_output,
        ))
    }

    #[inline]
    fn validate_circuit(
        &self,
        circuit: &Circuit,
        number_used_qubits: usize,
    ) -> Result<(), RoqoqoBackendError> {
        if number_used_qubits > self.number_qubits {
            return Err(RoqoqoBackendError::GenericError {
                msg: format!(
                    "Insufficient qubits in backend. \
                     Available qubits: {} \
                     Number of qubits used in circuit: {}",
                    self.number_qubits, number_used_qubits
                ),
            });
        }
        Ok(())
    }
}

/// Handler for repeated measurements.
///
/// The number of simulation repetitions set with PragmaSimulationRepetitions should be greater than
/// one if at least one of the following conditions if met
///
/// 1) There are multiple measure operations on the same qubit
/// 2) A qubit is acted upon after it has been measured
fn handle_repeated_measurements(
    circuit: &Circuit,
    repeated_measurement_readout: &mut Option<String>,
    number_measurements: &mut Option<usize>,
    repetitions: &mut usize,
    replacement_measurement: &mut Option<PragmaRepeatedMeasurement>,
    replace_measurements: &mut Option<usize>,
    register_lengths: &HashMap<String, usize>,
) -> Result<(), RoqoqoBackendError> {
    let mut measured_qubits: Vec<usize> = vec![];
    // let mut measured_qubits_in_repeated_measurement: Vec<usize> = vec![];
    let mut simulation_repetitions: Option<usize> = None;
    // actual number of repetitions for repeated measurements
    let mut effective_number_measurements: usize = 1;
    // flag for when PragmaSimulationRepetitions is needed
    let mut stochastic_simulation = false;
    // whether to rerun the whole circuit for every measurement of a repeated measurement, or to
    // just sample from the probability distribution before the measurement
    let mut rerun_whole_circuit: bool = false;

    for op in circuit.iter() {
        match op {
            Operation::PragmaRepeatedMeasurement(o) => match number_measurements {
                Some(_) => {
                    return Err(RoqoqoBackendError::GenericError {
                        msg: REPEATED_MEAS_ERROR.to_string(),
                    })
                }
                None => {
                    let number_qubits: usize =
                        *register_lengths
                            .get(o.readout())
                            .ok_or(RoqoqoBackendError::GenericError {
                            msg: "No register corresponding to PragmaRepeatedMeasurement readout \
                                found."
                                .to_string(),
                        })?;
                    let involved_qubits = (0..number_qubits - 1).collect::<Vec<usize>>();
                    // measured_qubits_in_repeated_measurement.extend(involved_qubits);
                    if involved_qubits.iter().any(|q| measured_qubits.contains(q)) {
                        rerun_whole_circuit = true;
                    }
                    measured_qubits.extend(involved_qubits);
                    number_measurements.replace(*o.number_measurements());
                    repeated_measurement_readout.replace(o.readout().clone());
                }
            },
            Operation::PragmaSetNumberOfMeasurements(o) => match number_measurements {
                Some(_) => {
                    return Err(RoqoqoBackendError::GenericError {
                        msg: REPEATED_MEAS_ERROR.to_string(),
                    })
                }
                None => {
                    number_measurements.replace(*o.number_measurements());
                    repeated_measurement_readout.replace(o.readout().clone());
                }
            },
            Operation::PragmaSimulationRepetitions(o) => {
                simulation_repetitions.replace(o.repetitions());
            }
            _ => (),
        }
    }

    for op in circuit.iter() {
        println!("{:?}", replace_measurements);
        match op {
            Operation::MeasureQubit(o) => {
                // if a qubit in a repeated measurement has already been measured, the circuit needs
                // to be rerun from the beginning for every repetition of the repeated measurement
                if let Some(rm_readout) = repeated_measurement_readout {
                    let out_reg = o.readout();
                    if out_reg == rm_readout {
                        println!("YEA");
                        rerun_whole_circuit = measured_qubits.contains(o.qubit());
                        replace_measurements.replace(*o.qubit());
                    }
                }
                measured_qubits.push(*o.qubit());
            }
            // TODO check: do we need to do this also for some operations that involve All qubits?
            _ => {
                // check if qubits are being acted upon after a measurement (condition 2 in the docstring)
                if let InvolvedQubits::Set(set) = op.involved_qubits() {
                    if set.iter().any(|q| measured_qubits.contains(q)) {
                        stochastic_simulation = true;
                        *replace_measurements = None;
                    }
                }
            }
        }
    }

    // Check if qubits are being measured more than once (condition 1 in the docstring)
    if contains_duplicates(measured_qubits) {
        println!("qubits are being measured multiple times");
        stochastic_simulation = true;
        rerun_whole_circuit = true;
    }

    // sanity checks
    if stochastic_simulation && simulation_repetitions.is_none() {
        return Err(RoqoqoBackendError::GenericError {
            msg:
                "Circuit requires a stochastic simulation, but a number of simulation repetitions \
                 is not set with PragmaSimulationRepetitions."
                    .to_string(),
        });
    } else if !stochastic_simulation && simulation_repetitions.is_some() {
        return Err(RoqoqoBackendError::GenericError {
            msg:
                "A number of simulation repetitions is set with PragmaSimulationRepetitions, but a \
                 stochastic simulation is not needed."
                    .to_string(),
        });
    }

    // The number of repeated measurements should be a multiple of the number of simulation repetitions,
    // so that the number of repeated measurements represents the number of shots on real hardware.
    // In the simulator, repeated measurements are executed (number_measurements /
    // simulation_repetitions) times, so that the shot noise of the results is still determined by
    // number_measurements.
    if let Some(sim_rep) = simulation_repetitions {
        if let Some(num_meas) = number_measurements.clone() {
            if num_meas % sim_rep != 0 {
                return Err(RoqoqoBackendError::GenericError {
                    msg:
                        "When both a repeated measurement and PragmaSimulationRepetitions are set, \
                         the number of repeated measurements needs to be a multiple of the number \
                         of simulation repetitions."
                            .to_string(),
                });
            } else if !rerun_whole_circuit {
                // Set the effective number of repetitions of the repeated measurements to their
                // ratio, so that the number of repeated measurements set by the user represents
                // the number of shots
                effective_number_measurements = num_meas / sim_rep;
                println!(
                    "Computed effective number measurememnts: {:?}",
                    effective_number_measurements
                );
            }
        }
    }

    // compute the number of times to simulate the whole circuit from the beginning
    match rerun_whole_circuit {
        // if true, run the whole circuit for the number of shots set by the user in the repeated measurement
        true => {
            if let Some(num_meas) = number_measurements {
                *repetitions = *num_meas;
            }
        }
        // if false, rerun the whole circuit for the number of times specified with PragmaSimulationRepetitions
        false => {
            *repetitions = simulation_repetitions.unwrap_or(1);
            *replacement_measurement = construct_replacement_measurement(
                circuit,
                *replace_measurements,
                repeated_measurement_readout,
                number_measurements,
            );
        }
    }
    number_measurements.replace(effective_number_measurements);

    println!("rerun_whole_circuit: {:?}", rerun_whole_circuit);
    println!(
        "repeated_measurement_readout: {:?}",
        repeated_measurement_readout
    );
    println!("simulation_repetitions: {:?}", simulation_repetitions);

    Ok(())
}

#[inline]
fn construct_replacement_measurement(
    circuit: &Circuit,
    replace_measurements: Option<usize>,
    repeated_measurement_readout: &Option<String>,
    number_measurements: &Option<usize>,
) -> Option<PragmaRepeatedMeasurement> {
    if replace_measurements.is_some() {
        let name = repeated_measurement_readout.clone().expect(
            "Internal bug: no repeated measurement readout found when constructing \
                     replacement measurement.",
        );
        let number_measurements = number_measurements.expect(
            "Internal bug: no number_measurements found when constructing \
                     replacement measurement.",
        );
        let mut reordering_map: HashMap<usize, usize> = HashMap::new();
        for op in circuit.iter() {
            if let Operation::MeasureQubit(measure) = op {
                reordering_map.insert(*measure.qubit(), *measure.readout_index());
            }
        }
        Some(PragmaRepeatedMeasurement::new(
            name,
            number_measurements,
            Some(reordering_map),
        ))
    } else {
        None
    }
}

type InternalRegisters<'a> = (
    &'a mut HashMap<String, Vec<bool>>,
    &'a mut HashMap<String, Vec<f64>>,
    &'a mut HashMap<String, Vec<num_complex::Complex<f64>>>,
);

// groups replace_measurements and repeated_measurement_pragma
type ReplacedMeasurementInformation<'a> = (Option<usize>, &'a Option<PragmaRepeatedMeasurement>);

fn run_inner_circuit_loop(
    register_lengths: &HashMap<String, usize>,
    circuit: &Circuit,
    replaced_measurement_information: ReplacedMeasurementInformation,
    qureg: &mut Qureg,
    registers_internal: InternalRegisters,
    bit_registers_output: &mut HashMap<String, Vec<Vec<bool>>>,
    device: &mut Option<Box<dyn roqoqo::devices::Device>>,
    number_measurements: Option<usize>,
) -> Result<(), RoqoqoBackendError> {
    let (replace_measurements, replacement_measurement) = replaced_measurement_information;
    let (bit_registers_internal, float_registers_internal, complex_registers_internal) =
        registers_internal;
    let number_measurements = number_measurements.expect(
        "Internal bug: number measurements should not be None if a repeated \
                             measurement is present.",
    );

    for op in circuit.iter() {
        println!("\nINNER LOOP: executing {:?}\n", op.clone());
        match op {
            Operation::PragmaRepeatedMeasurement(rm) => match replace_measurements {
                None => {
                    println!("Replacing repeated pragma with measureQubit");
                    let number_qubits: usize = register_lengths
                        .get(rm.readout())
                        .expect("Internal bug: Register for repeated measurement not found.")
                        .to_owned();
                    for qb in 0..number_qubits {
                        let ro_index = match rm.qubit_mapping() {
                            Some(mp) => mp.get(&qb).unwrap_or(&qb),
                            None => &qb,
                        };
                        let mqb_new: Operation =
                            MeasureQubit::new(qb, rm.readout().to_owned(), *ro_index).into();
                        call_operation_with_device(
                            &mqb_new,
                            qureg,
                            bit_registers_internal,
                            float_registers_internal,
                            complex_registers_internal,
                            bit_registers_output,
                            device,
                        )?;
                    }
                }
                Some(_) => {
                    execute_pragma_repeated_measurement(
                        rm,
                        qureg,
                        bit_registers_internal,
                        bit_registers_output,
                        number_measurements,
                    )?;
                }
            },
            Operation::MeasureQubit(internal_op) => {
                if let Some(position) = replace_measurements {
                    println!("This MeasureQubit is part of a replaced measurement");
                    if internal_op.qubit() == &position {
                        if let Some(helper) = replacement_measurement.as_ref() {
                            println!("Executing replaced measurement...\n");
                            execute_pragma_repeated_measurement(
                                helper,
                                qureg,
                                bit_registers_internal,
                                bit_registers_output,
                                number_measurements,
                            )?;
                        }
                    }
                } else {
                    println!("Calling MeasureQubit directly");
                    call_operation_with_device(
                        op,
                        qureg,
                        bit_registers_internal,
                        float_registers_internal,
                        complex_registers_internal,
                        bit_registers_output,
                        device,
                    )?;
                }
            }
            _ => {
                call_operation_with_device(
                    op,
                    qureg,
                    bit_registers_internal,
                    float_registers_internal,
                    complex_registers_internal,
                    bit_registers_output,
                    device,
                )?;
            }
        }
    }
    Ok(())
}

#[inline]
fn find_pragma_op(op: &Operation) -> bool {
    match op {
        Operation::PragmaConditional(x) => x.circuit().iter().any(|x| find_pragma_op(&x)),
        Operation::PragmaLoop(x) => x.circuit().iter().any(|x| find_pragma_op(&x)),
        Operation::PragmaGetPauliProduct(x) => x.circuit().iter().any(|x| find_pragma_op(&x)),
        Operation::PragmaGetOccupationProbability(x) => {
            if let Some(circ) = x.circuit() {
                circ.iter().any(|x| find_pragma_op(&x))
            } else {
                false
            }
        }
        Operation::PragmaGetDensityMatrix(x) => {
            if let Some(circ) = x.circuit() {
                circ.iter().any(|x| find_pragma_op(&x))
            } else {
                false
            }
        }
        Operation::PragmaDamping(_)
        | Operation::PragmaDephasing(_)
        | Operation::PragmaDepolarising(_)
        | Operation::PragmaGeneralNoise(_)
        | Operation::PragmaSetDensityMatrix(_) => true,
        _ => false,
    }
}

// check if a vector contains duplicates
#[inline]
fn contains_duplicates(vec: Vec<usize>) -> bool {
    let mut seen = HashSet::new();
    for &value in &vec {
        if !seen.insert(value) {
            return true;
        }
    }
    false
}

#[cfg(feature = "async")]
#[async_trait]
impl AsyncEvaluatingBackend for Backend {
    async fn async_run_circuit_iterator<'a>(
        &self,
        circuit: impl Iterator<Item = &'a Operation> + std::marker::Send,
    ) -> RegisterResult {
        self.run_circuit_iterator(circuit)
    }

    #[cfg(feature = "parallelization")]
    async fn async_run_measurement_registers<T>(&self, measurement: &T) -> RegisterResult
    where
        T: Measure,
        T: std::marker::Sync,
    {
        self.run_measurement_registers(measurement)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qoqo_calculator::CalculatorFloat;

    #[test]
    fn test_find_pragma_op() {
        let op = roqoqo::operations::Operation::from(roqoqo::operations::PragmaConditional::new(
            "bits".to_owned(),
            0,
            vec![Operation::from(roqoqo::operations::PragmaDamping::new(
                0,
                CalculatorFloat::PI,
                CalculatorFloat::ZERO,
            ))]
            .into_iter()
            .collect(),
        ));
        assert!(find_pragma_op(&&op));

        let op = roqoqo::operations::Operation::from(roqoqo::operations::PragmaLoop::new(
            CalculatorFloat::from(5),
            vec![Operation::from(roqoqo::operations::PragmaDephasing::new(
                1,
                CalculatorFloat::PI,
                CalculatorFloat::ZERO,
            ))]
            .into_iter()
            .collect(),
        ));
        assert!(find_pragma_op(&&op));

        let op =
            roqoqo::operations::Operation::from(roqoqo::operations::PragmaGetPauliProduct::new(
                HashMap::new(),
                "pauli".to_owned(),
                vec![Operation::from(
                    roqoqo::operations::PragmaDepolarising::new(
                        1,
                        CalculatorFloat::PI,
                        CalculatorFloat::ZERO,
                    ),
                )]
                .into_iter()
                .collect(),
            ));
        assert!(find_pragma_op(&&op));

        let op = roqoqo::operations::Operation::from(
            roqoqo::operations::PragmaGetOccupationProbability::new(
                "float_register".to_owned(),
                Some(
                    vec![Operation::from(
                        roqoqo::operations::PragmaGeneralNoise::new(
                            1,
                            CalculatorFloat::PI,
                            ndarray::array![[0.], [1.]],
                        ),
                    )]
                    .into_iter()
                    .collect(),
                ),
            ),
        );
        assert!(find_pragma_op(&&op));

        let op =
            roqoqo::operations::Operation::from(roqoqo::operations::PragmaGetDensityMatrix::new(
                "complex_register".to_owned(),
                Some(
                    vec![Operation::from(
                        roqoqo::operations::PragmaSetDensityMatrix::new(ndarray::array![
                            [num_complex::Complex::new(1., 0.)],
                            [num_complex::Complex::new(0., 1.)]
                        ]),
                    )]
                    .into_iter()
                    .collect(),
                ),
            ));
        assert!(find_pragma_op(&&op));

        let op = roqoqo::operations::Operation::from(
            roqoqo::operations::PragmaGetDensityMatrix::new("complex_register".to_owned(), None),
        );
        assert!(!find_pragma_op(&&op));
    }
}
