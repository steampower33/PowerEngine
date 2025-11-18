#include "context.h"
#include "graphics_context.h"

#include "skybox.h"

Skybox::Skybox(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext)
{
    mesh_data_ = std::move(meshData);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.vertices, mesh_data_.vertex_buffer, mesh_data_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.indices, mesh_data_.index_buffer, mesh_data_.index_buffer_memory);

    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale_);
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation_);
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

    world_ = translationMatrix * rotationMatrix * scaleMatrix;
}
