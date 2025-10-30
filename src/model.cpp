#include "vertex.h"
#include "camera.h"

#include "model.h"

Model::Model(const std::string modelPath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool)
{
    LoadModel(modelPath);

    vku::CreateVertexBuffer(physicalDevice, device, queue, commandPool, vertices_, vertex_buffer_, vertex_buffer_memory_);
    vku::CreateIndexBuffer(physicalDevice, device, queue, commandPool, indices_, index_buffer_, index_buffer_memory_);
}

void Model::LoadModel(const std::string& modelPath) {
    // Use tinygltf to load the model instead of tinyobjloader
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelPath);

    if (!warn.empty()) {
        std::cout << "glTF warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "glTF error: " << err << std::endl;
    }

    if (!ret) {
        throw std::runtime_error("Failed to load glTF model");
    }

    // Process all meshes in the model
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            // Get indices
            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

            // Get vertex positions
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

            // Get texture coordinates if available
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            const tinygltf::Accessor* texCoordAccessor = nullptr;
            const tinygltf::BufferView* texCoordBufferView = nullptr;
            const tinygltf::Buffer* texCoordBuffer = nullptr;

            if (hasTexCoords) {
                texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
                texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
            }

            //// Get normals if available
            //bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            //const tinygltf::Accessor* normalAccessor = nullptr;
            //const tinygltf::BufferView* normalBufferView = nullptr;
            //const tinygltf::Buffer* normalBuffer = nullptr;

            //if (hasNormals) {
            //    normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];
            //    normalBufferView = &model.bufferViews[normalAccessor->bufferView];
            //    normalBuffer = &model.buffers[normalBufferView->buffer];
            //}

            // Process vertices
            for (size_t i = 0; i < posAccessor.count; i++) {
                Vertex vertex{};

                // Get position
                const float* pos = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * 12]);
                vertex.pos = { pos[0], pos[1], pos[2] };

                // Get texture coordinates if available
                if (hasTexCoords) {
                    const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * 8]);
                    vertex.texcoord = { texCoord[0], 1.0f - texCoord[1] };
                }
                else {
                    vertex.texcoord = { 0.0f, 0.0f };
                }

                //// Get normal if available
                //if (hasNormals) {
                //    // ���� ���ʹ� float 3�� (12����Ʈ)
                //    const float* n = reinterpret_cast<const float*>(&normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * sizeof(glm::vec3)]);
                //    vertex.normal = { n[0], n[1], n[2] };
                //}
                //else {
                //    // �𵨿� ���� �����Ͱ� ���� ���, �⺻��(�Ǵ� ���߿� ���)
                //    vertex.normal = { 0.0f, 0.0f, 0.0f };
                //}

                // Add vertex if unique
                if (!uniqueVertices.contains(vertex)) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices_.size());
                    vertices_.push_back(vertex);
                }
            }

            // Process indices
            const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];

            // Handle different index component types
            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const uint16_t* indices16 = reinterpret_cast<const uint16_t*>(indexData);
                for (size_t i = 0; i < indexAccessor.count; i++) {
                    Vertex vertex = vertices_[indices16[i]];
                    indices_.push_back(uniqueVertices[vertex]);
                }
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const uint32_t* indices32 = reinterpret_cast<const uint32_t*>(indexData);
                for (size_t i = 0; i < indexAccessor.count; i++) {
                    Vertex vertex = vertices_[indices32[i]];
                    indices_.push_back(uniqueVertices[vertex]);
                }
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                const uint8_t* indices8 = reinterpret_cast<const uint8_t*>(indexData);
                for (size_t i = 0; i < indexAccessor.count; i++) {
                    Vertex vertex = vertices_[indices8[i]];
                    indices_.push_back(uniqueVertices[vertex]);
                }
            }
        }
    }
}

void Model::ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta)
{   // 1. ������Ʈ�� ���� ������Ʈ�Ѵ�.

    // ���� ��ġ�� �̵����� ���Ѵ�.
    position_ += translationDelta;

    // ���� ȸ���� ���ο� ȸ���� '�տ�' �����ش�.
    // (q_new * q_old �� old ȸ�� �� new ȸ���� �����ϴ� �Ͱ� ����)
    rotation_ = rotationDelta * rotation_;

    // ���ʹϾ��� �ε��Ҽ��� ������ ���̰� 1�� �ƴϰ� �� �� �����Ƿ�,
    // �ֱ������� ����ȭ���ִ� ���� ����.
    rotation_ = glm::normalize(rotation_);

    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale_);
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation_); // ���ʹϾ� -> ȸ�� ���
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

    // 2. SRT (Scale -> Rotate -> Translate) ������ �����Ͽ� ���� ���� ����� ����Ѵ�.
    world_ = translationMatrix * rotationMatrix * scaleMatrix;
}

void Model::UpdateUBO(Camera& camera, glm::vec2 viewportSize, uint32_t currentFrame)
{
    uniform_data.model = world_;
    uniform_data.view = camera.View();
    uniform_data.proj = camera.Proj(viewportSize.x, viewportSize.y);

    memcpy(uniform_buffers_mapped[currentFrame], &uniform_data, sizeof(uniform_data));

}