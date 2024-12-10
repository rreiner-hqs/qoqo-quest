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
    call_operation_with_device, execute_pragma_repeated_measurement,
    execute_replaced_repeated_measurement, initialize_registers,
};

use roqoqo::backends::EvaluatingBackend;
use roqoqo::backends::RegisterResult;
use roqoqo::operations::*;
use roqoqo::registers::{
    BitOutputRegister, BitRegister, ComplexOutputRegister, ComplexRegister, FloatOutputRegister,
    FloatRegister,
};
use roqoqo::Circuit;
use roqoqo::RoqoqoBackendError;

use crate::Qureg;
use std::collections::HashMap;

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
    /// `number_qubits` - The number of qubits supported by the backend
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
    /// `random_seed` - The random seed to use for the backend
    pub fn set_random_seed(&mut self, random_seed: Vec<u64>) {
        self.random_seed = Some(random_seed);
    }

    /// Gets the current random seed set for the backend.
    ///
    /// # Returns
    ///
    /// `Option<Vec<u64>>` - The current random seed
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
        // TODO Is this needed?
        // let circuit_vec: Vec<&'a Operation> = circuit.into_iter().collect();

        // Initialize output register names from circuit definitions
        let (output_registers, register_lengths, number_used_qubits) =
            initialize_registers(&circuit)?;

        self.validate_circuit(&circuit, number_used_qubits)?;

        let (bit_registers_output, float_registers_output, complex_registers_output) =
            output_registers;
        let (bit_registers_lengths, float_registers_lengths, complex_registers_lengths) =
            register_lengths;

        // Automatically switch to density matrix mode if operations are present in the
        // circuit that require density matrix mode
        let is_density_matrix = circuit.iter().any(find_pragma_op);

        // TODO not used at the moment. We would need to adjust all the tests in the other HQS
        // modules if we accounted for global phases.
        // let global_phase = circuit_vec
        //     .iter()
        //     .filter_map(|x| match x {
        //         Operation::PragmaGlobalPhase(x) => Some(x.phase()),
        //         _ => None,
        //     })
        //     .fold(CalculatorFloat::ZERO, |acc, x| acc + x);

        // Number of measurements as set by the repeated measurement operation (PragmaRepeatedMeasurement
        // or PragmaSetNumberOfMeasurements), if present
        let mut number_measurements: Option<usize> = None;
        // Readout register name for the repeated measurement operation (PragmaRepeatedMeasurement
        // or PragmaSetNumberOfMeasurements), if present
        let mut repeated_measurement_readout: Option<String> = None;

        handle_repeated_measurements(
            &circuit,
            &mut number_measurements,
            &mut repeated_measurement_readout,
        )?;

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

        for _ in 0..repetitions {
            qureg.reset();
            let mut bit_registers_internal: HashMap<String, BitRegister> = HashMap::new();
            let mut float_registers_internal: HashMap<String, FloatRegister> = HashMap::new();
            let mut complex_registers_internal: HashMap<String, ComplexRegister> = HashMap::new();
            run_inner_circuit_loop(
                &bit_registers_lengths,
                &circuit,
                &mut qureg,
                (
                    &mut bit_registers_internal,
                    &mut float_registers_internal,
                    &mut complex_registers_internal,
                ),
                &mut bit_registers_output,
                device,
            )?;

            // Append bit result of one circuit execution to output register
            for (name, register) in bit_registers_output.iter_mut() {
                if let Some(tmp_reg) = bit_registers_internal.get(name) {
                    if name != &repeated_measurement_readout {
                        register.push(tmp_reg.to_owned())
                    }
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
/// Scans the circuit to either PragmaRepeatedMeasurement or PragmaSetNumberOfMeasurements, and
/// saves the number of measurements as well as the name of the readout register.
#[inline]
fn handle_repeated_measurements(
    circuit: &Circuit,
    number_measurements: &mut Option<usize>,
    repeated_measurement_readout: &mut Option<String>,
) -> Result<(), RoqoqoBackendError> {
    for op in circuit.iter() {
        match op {
            Operation::PragmaRepeatedMeasurement(o) => match number_measurements {
                Some(_) => {
                    return Err(RoqoqoBackendError::GenericError {
                        msg: REPEATED_MEAS_ERROR.to_string(),
                    })
                }
                None => {
                    *number_measurements = Some(*o.number_measurements());
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
                    *number_measurements = Some(*o.number_measurements());
                    repeated_measurement_readout.replace(o.readout().clone());
                }
            },
            _ => (),
        }
    }

    let found_fitting_measurement = if let Some(readout_name) = repeated_measurement_readout {
        circuit.iter().any(|op| match op {
            Operation::MeasureQubit(inner_op) => inner_op.readout() == readout_name,
            Operation::PragmaRepeatedMeasurement(inner_op) => {
                inner_op.readout() == readout_name
            }
            _ => false,
        })
    } else {
        true
    };

    if !found_fitting_measurement {
        if let Some(readout_name) = repeated_measurement_readout {
            return Err(RoqoqoBackendError::GenericError {
                msg: format!(
                    "No matching measurement found for PragmaSetNumberOfMeasurements for readout `{}`",
                    readout_name
                ),
            });
        }
    }

    return Ok(());
}

type InternalRegisters<'a> = (
    &'a mut HashMap<String, Vec<bool>>,
    &'a mut HashMap<String, Vec<f64>>,
    &'a mut HashMap<String, Vec<num_complex::Complex<f64>>>,
);

fn run_inner_circuit_loop(
    register_lengths: &HashMap<String, usize>,
    circuit: &Circuit,
    qureg: &mut Qureg,
    registers_internal: InternalRegisters,
    bit_registers_output: &mut HashMap<String, Vec<Vec<bool>>>,
    device: &mut Option<Box<dyn roqoqo::devices::Device>>,
) -> Result<(), RoqoqoBackendError> {
    let (bit_registers_internal, float_registers_internal, complex_registers_internal) =
        registers_internal;

    for op in circuit.iter() {
        match op {
            Operation::PragmaRepeatedMeasurement(rm) => {
                let number_qubits: usize = match register_lengths.get(rm.readout()) {
                    Some(pragma_nm) => {
                        let n = *pragma_nm;
                        n - 1
                    }
                    None => {
                        return Err(RoqoqoBackendError::GenericError {
                            msg: "No register corresponding to PragmaRepeatedMeasurement readout \
                                found"
                                .to_string(),
                        });
                    }
                };
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
