// @file im2row.hpp
// @brief Stack image patches as matrix rows
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__im2row5D__
#define __vl__im2row5D__

#include "../data.hpp"
#include <stddef.h>

namespace vl { namespace impl {

  template<vl::DeviceType dev, typename type>
  struct im2row5D {

    static vl::ErrorCode
    forward(vl::Context& context,
            type* stacked,
            type const* data,
            size_t height, size_t width, size_t depthlegth, size_t depth,
            size_t windowHeight, size_t windowWidth, size_t windowDepth,
            size_t strideY, size_t strideX, size_t strideD,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight,
            size_t padIn, size_t padOut) ;

    static vl::ErrorCode
    backward(vl::Context& context,
             type* data,
             type const* stacked,
             size_t height, size_t width, size_t depthlegth, size_t depth,
             size_t windowHeight, size_t windowWidth, size_t windowDepth,
             size_t strideY, size_t strideX, size_t strideD,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight,
             size_t padIn, size_t padOut) ;
  } ;

} }

#endif /* defined(__vl__im2row__) */
