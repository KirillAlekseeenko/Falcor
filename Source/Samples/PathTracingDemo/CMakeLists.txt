add_falcor_executable(PathTracingDemo)

target_sources(PathTracingDemo PRIVATE
    PathTracingDemo.cpp
    PathTracingDemo.h
    RenderShadowMap.cpp
    RenderShadowMap.h
    DiffusePathTracer.cpp
    DiffusePathTracer.h
    BrightnessScale.ps.slang
    DiffusePathTracer.rt.slang
    ShadowPass.3d.slang
    SimpleRasterPass.3d.slang
)

target_copy_shaders(PathTracingDemo Samples/PathTracingDemo)

target_source_group(PathTracingDemo "Samples")
