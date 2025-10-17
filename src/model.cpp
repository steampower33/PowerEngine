#include "model.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Model::Model(const std::string modelPath, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& queue)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (!uniqueVertices.contains(vertex)) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices_.size());
                vertices_.push_back(vertex);
            }

            indices_.push_back(uniqueVertices[vertex]);
        }
    }

    createVertexBuffer(device, physicalDevice, commandPool, queue, vertices_, vertexBuffer_, vertexBufferMemory_);
    createIndexBuffer(device, physicalDevice, commandPool, queue, indices_, indexBuffer_, indexBufferMemory_);
}

Model::~Model()
{

}


