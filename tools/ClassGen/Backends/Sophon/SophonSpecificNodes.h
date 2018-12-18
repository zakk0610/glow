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

BB.newBackendSpecificNode("SophonConv")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "KernelsHW")
    .addMember(MemberType::VectorUnsigned, "StridesHW")
    .addMember(MemberType::VectorUnsigned, "PadsTBLR")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Unsigned, "RShift")
    .addResultFromCtorArg()
    .setDocstring("This is a sophon ConvolutionQ8 implementation");

BB.includeBackendSpecificVerification("glow/SophonSpecificNodesVerification.h");

#endif // GLOW_WITH_SOPHON
