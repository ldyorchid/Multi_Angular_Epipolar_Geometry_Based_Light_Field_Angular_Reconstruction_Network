// @file nnconv5D.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-16 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnconv5D.hpp"
#include "nnbias.hpp"
#include "impl/nnconv5D_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnconv5D_cudnn.hpp"
#endif
#include <assert.h>
#include <iostream>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

/*
 for output: must have data and optional filters or biases
 */

#define DISPATCH(deviceType, dataType) \
error = vl::impl::nnconv5D_forward_blas<deviceType, dataType> \
(context, \
output, outputMult, \
data, dataMult, \
filters, biases, \
 strideY, strideX, strideZ,\
 padTop, padBottom, \
 padLeft, padRight, \
 padIn, padOut) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnconv5D_cudnn<dataType>::forward \
(context, \
 output, outputMult, \
 data, dataMult, \
 filters, biases, \
 strideY, strideX, strideZ\
 padTop, padBottom, \
 padLeft, padRight, \
 padIn, padOut) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnconv5D_forward(Context& context,
                   Tensor output, double outputMult,
                   Tensor data, double dataMult,
                   Tensor filters,
                   Tensor biases,
                   int strideY, int strideX, int strideZ,
                   int padTop, int padBottom,
                   int padLeft, int padRight,
                   int padIn, int padOut)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

   switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (error == vl::VLE_Success) { return error ; }
        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      std::cout<<"PASS DISPATCH BLAS";
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv5D_backward */
/* ---------------------------------------------------------------- */


/*
 for derBiases:  must have derOuptut
 for derData:    must have derData, derOutput and filters
 for derFilters: must have derFilters, derOutput and data
 */

#undef DISPATCH
#define DISPATCH(deviceType, dataType) \
error = vl::impl::nnconv5D_backward_blas<deviceType, dataType> \
(context, \
 derData, derFilters, derBiases, \
 data, filters, derOutput, \
 strideY, strideX, strideZ, \
 padTop, padBottom, \
 padLeft, padRight, \
 padIn, padOut) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnconv5D_cudnn<dataType>::backward \
(context, \
 derData, derFilters, derBiases, \
 data, filters, derOutput, \
 strideY, strideX, strideZ,\
 padTop, padBottom, \
 padLeft, padRight, \
 padIn, padOut) ;

vl::ErrorCode
vl::nnconv5D_backward(Context& context,
                    Tensor derData,
                    Tensor derFilters,
                    Tensor derBiases,
                    Tensor data,
                    Tensor filters,
                    Tensor derOutput,
                    int strideY, int strideX, int strideZ,
                   int padTop, int padBottom,
                   int padLeft, int padRight,
                   int padIn, int padOut)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;
  
  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (error == vl::VLE_Success) { return error ; }
        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return error ;
}
