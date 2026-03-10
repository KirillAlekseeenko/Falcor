add_plugin(NewSimpleRenderPass)

target_sources(NewSimpleRenderPass PRIVATE
    NewSimpleRenderPass.cpp
    NewSimpleRenderPass.h
)

target_source_group(NewSimpleRenderPass "RenderPasses")
