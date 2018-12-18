/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef GLOW_WITH_SOPHON

#include "glow/Graph/VerifierHelper.h"

bool SophonConvNode::verify() const {
  bool verify = true;
  verify |= checkSameType(getInput(), getResult(), this);
  verify |= expectCompareTrue("Invalid input type",
                              getInput().getType()->getElementType(),
                              ElemKind::Int8QTy, this);
  verify |= expectCompareTrue("Invalid Filter type",
                              getFilter().getType()->getElementType(),
                              ElemKind::Int8QTy, this);
  // bias must be a 16-bit tensor
  verify |= expectCompareTrue("Invalid Bias type",
                              getBias().getType()->getElementType(),
                              ElemKind::Int16QTy, this);
  return verify;
}

#endif // GLOW_WITH_SOPHON
