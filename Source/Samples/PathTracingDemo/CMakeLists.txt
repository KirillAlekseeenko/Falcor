add_falcor_executable(PathTracingDemo)

target_sources(PathTracingDemo PRIVATE
    PathTracingDemo.cpp
    PathTracingDemo.h
    RenderShadowMap.cpp
    RenderShadowMap.h
)

target_copy_shaders(PathTracingDemo Samples/PathTracingDemo)

target_source_group(PathTracingDemo "Samples")
