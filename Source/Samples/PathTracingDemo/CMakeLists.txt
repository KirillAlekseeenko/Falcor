add_falcor_executable(PathTracingDemo)

target_sources(PathTracingDemo PRIVATE
    PathTracingDemo.cpp
    PathTracingDemo.h
)

target_copy_shaders(PathTracingDemo Samples/PathTracingDemo)

target_source_group(PathTracingDemo "Samples")
