add_falcor_executable(IrradianceSamplesBaker)

target_sources(IrradianceSamplesBaker PRIVATE
    IrradianceSampleDebugVis.cpp
    IrradianceSampleDebugVis.h
    IrradianceSamplesBaker.cpp
    IrradianceSamplesBaker.h
    IrradianceSamplesBaker.3d.slang
    IrradianceSamplesBaker.rt.slang
    DebugSpheres.3d.slang
)

target_copy_shaders(IrradianceSamplesBaker Samples/IrradianceSamplesBaker)

target_source_group(IrradianceSamplesBaker "Samples")
