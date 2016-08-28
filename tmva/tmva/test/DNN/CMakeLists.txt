############################################################################
# CMakeLists.txt file for building TMVA/DNN tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

#--- CUDA tests. ---------------------------
find_package(CUDA)
if (CUDA_FOUND)

  SET(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})

  # DNN - Activation Functions Cuda
  CUDA_ADD_EXECUTABLE(testActivationFunctionsCuda TestActivationFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testActivationFunctionsCuda ${LIBRARIES} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ActivationFunctionsCuda COMMAND testDerivativesCuda)

  # DNN - Loss Functions Cuda
  CUDA_ADD_EXECUTABLE(testLossFunctionsCuda TestLossFunctionsCuda.cxx)
  TARGET_LINK_LIBRARIES(testLossFunctionsCuda ${LIBRARIES} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-LossFunctionsCuda COMMAND testLossFunctionsCuda)

  # DNN - Derivatives Cuda
  CUDA_ADD_EXECUTABLE(testDerivativesCuda TestDerivativesCuda.cxx)
  TARGET_LINK_LIBRARIES(testDerivativesCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-DerivativesCuda COMMAND testDerivativesCuda)

  # DNN - Backpropagation Cuda
  CUDA_ADD_EXECUTABLE(testBackpropagationCuda TestBackpropagationCuda.cxx)
  TARGET_LINK_LIBRARIES(testBackpropagationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-BackpropagationCuda COMMAND testBackpropagationCuda)

  # DNN - Minimization Cuda
  CUDA_ADD_EXECUTABLE(testMinimizationCuda TestMinimizationCuda.cxx)
  TARGET_LINK_LIBRARIES(testMinimizationCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-MinimizationCuda COMMAND testMinimizationCuda)

  # DNN - Arithmetic Cuda
  CUDA_ADD_EXECUTABLE(testArithmeticCuda TestMatrixArithmeticCuda.cxx)
  TARGET_LINK_LIBRARIES(testArithmeticCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ArithmeticCuda COMMAND testArithmeticCuda)

  # DNN - DataLoader Cuda
  CUDA_ADD_EXECUTABLE(testDataLoaderCuda TestDataLoaderCuda.cxx)
  TARGET_LINK_LIBRARIES(testDataLoaderCuda ${Libraries} ${DNN_CUDA_LIBRARIES})
endif (CUDA_FOUND)

#--- CPU tests. ----------------------------
find_package(BLAS)
if (BLAS_FOUND AND imt)

  # DNN - Arithmetic Functions CPU
  ROOT_EXECUTABLE(testArithmeticCpu TestMatrixArithmeticCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Arithmetic-Cpu COMMAND testArithmeticCpu)

  # DNN - Activation Functions CPU
  ROOT_EXECUTABLE(testActivationFunctionsCpu TestActivationFunctionsCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Activation-Functions-Cpu COMMAND testActivationFunctionsCpu)

  # DNN - Loss Functions CPU
  ROOT_EXECUTABLE(testLossFunctionsCpu TestLossFunctionsCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Loss-Functions-Cpu COMMAND testLossFunctionsCpu)

  # DNN - Derivatives CPU
  ROOT_EXECUTABLE(testDerivativesCpu TestDerivativesCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Derivatives-Cpu COMMAND testDerivativesCpu)

  # DNN - Backpropagation CPU
  ROOT_EXECUTABLE(testBackpropagationCpu TestBackpropagationCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Backpropagation-Cpu COMMAND testBackpropagationCpu)

  # DNN - DataLoader CPU
  ROOT_EXECUTABLE(testDataLoaderCpu TestDataLoaderCpu.cxx
    LIBRARIES ${Libraries})
  ROOT_ADD_TEST(TMVA-DNN-Data-Loader-Cpu COMMAND testDataLoaderCpu)

  # DNN - Minimization CPU
  ROOT_EXECUTABLE(testMinimizationCpu TestMinimizationCpu.cxx
    LIBRARIES ${Libraries} ${BLAS_openblas_LIBRARY} tbb)
  ROOT_ADD_TEST(TMVA-DNN-Minimization-Cpu COMMAND testMinimizationCpu)

endif (BLAS_FOUND AND imt)

#--- OpenCL tests -----------------
find_package(OpenCL)
if (OpenCL_FOUND AND clBLAS_DIR AND clRNG_DIR)

  INCLUDE_DIRECTORIES(${ROOT_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR}
    ${clBLAS_DIR}/src ${clRNG_DIR}/src/include)
  LINK_DIRECTORIES(${clBLAS_DIR}/library ${clRNG_DIR}/src/library)
  SET(DNN_OPENCL_LIBRARIES dnn_opencl ${OpenCL_LIBRARIES} clBLAS clRNG)

  # DNN - Matrix Arithmetic OpenCL
  LINK_DIRECTORIES(/home/simon/src/clBLAS/library)
  ROOT_EXECUTABLE(testMatrixArithmeticOpenCL TestMatrixArithmeticOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-MatrixArithmeticOpenCL COMMAND testMatrixArithmeticOpenCL)

  # DNN - Loss Functions OpenCL
  ROOT_EXECUTABLE(testLossFunctionsOpenCL TestLossFunctionsOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-LossFunctionsOpenCL COMMAND testLossFunctionsOpenCL)

  # DNN - Activation Functions OpenCL
  ROOT_EXECUTABLE(testActivationFunctionsOpenCL TestActivationFunctionsOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-ActivationFunctionsOpenCL COMMAND testActivationFunctionsOpenCL)
  # DNN - Derivatives OpenCL
  ROOT_EXECUTABLE(testDerivativesOpenCL TestDerivativesOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-DerivativesOpenCL COMMAND testDerivativesOpenCL)

  # DNN - Derivatives OpenCL
  ROOT_EXECUTABLE(testBackpropagationOpenCL TestBackpropagationOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-BackpropagationOpenCL COMMAND testBackPropagationOpenCL)

  # DNN - Dataloader OpenCL
  ROOT_EXECUTABLE(testDataLoaderOpenCL TestDataLoaderOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-TestDataLoaderOpenCL COMMAND testDataLoaderOpenCL)

  # DNN - Minimization OpenCL
  ROOT_EXECUTABLE(testMinimizationOpenCL TestMinimizationOpenCL.cxx
                  LIBRARIES ${Libraries} ${DNN_OPENCL_LIBRARIES})
  ROOT_ADD_TEST(TMVA-DNN-TestMinimizationOpenCL COMMAND testMinimizationOpenCL)

endif(OpenCL_FOUND AND clBLAS_DIR AND clRNG_DIR)
