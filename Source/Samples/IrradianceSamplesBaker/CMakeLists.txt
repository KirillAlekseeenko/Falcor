add_falcor_executable(IrradianceSamplesBaker)

target_sources(IrradianceSamplesBaker PRIVATE
    IrradianceSamplesBaker.cpp
    IrradianceSamplesBaker.h
    IrradianceSamplesBaker.rt.slang
)

target_copy_shaders(IrradianceSamplesBaker Samples/IrradianceSamplesBaker)

target_source_group(IrradianceSamplesBaker "Samples")
