
#include "context.h"
#include "graphics_context.h"
#include "vertex.h"
#include "camera.h"

#include "model.h"

Model::Model(const std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble)
{
    LoadModel(modelPath, vertexIncludeInfo);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.vertices, mesh_data_.vertex_buffer, mesh_data_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.indices, mesh_data_.index_buffer, mesh_data_.index_buffer_memory);

    model_count++;
    color_use = colorUse;
    moveble_ = moveble;

    ApplyTransform(initRotation, initPos);
}

Model::Model(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble)
{
    mesh_data_ = std::move(meshData);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.vertices, mesh_data_.vertex_buffer, mesh_data_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.indices, mesh_data_.index_buffer, mesh_data_.index_buffer_memory);

    model_count++;

    color_use = colorUse;
    moveble_ = moveble;

    ApplyTransform(initRotation, initPos);
}

void Model::LoadModel(const std::string& modelPath, const vku::VertexIncludeInfo& vertexIncludeInfo)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelPath);
    if (!ret) throw std::runtime_error("Failed to load glTF model");

    mesh_data_.vertices.clear();
    mesh_data_.indices.clear();

    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            // 1) geometry attribute 존재 여부 확인
            bool hasUV = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            bool hasTangents = primitive.attributes.find("TANGENT") != primitive.attributes.end();

            // 2) VertexIncludeInfo 까지 결합
            bool loadNormal = vertexIncludeInfo.normal && hasNormals;
            bool loadTangent = vertexIncludeInfo.tangent && hasTangents;

            // 3) Accessors
            const auto& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const auto& posBufferView = model.bufferViews[posAccessor.bufferView];
            const auto& posBuffer = model.buffers[posBufferView.buffer];

            const tinygltf::Accessor* uvAccessor = nullptr;
            const tinygltf::BufferView* uvBufferView = nullptr;
            const tinygltf::Buffer* uvBuffer = nullptr;

            if (hasUV)
            {
                uvAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                uvBufferView = &model.bufferViews[uvAccessor->bufferView];
                uvBuffer = &model.buffers[uvBufferView->buffer];
            }

            const tinygltf::Accessor* normalAccessor = nullptr;
            const tinygltf::BufferView* normalBufferView = nullptr;
            const tinygltf::Buffer* normalBuffer = nullptr;

            if (loadNormal)
            {
                normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];
                normalBufferView = &model.bufferViews[normalAccessor->bufferView];
                normalBuffer = &model.buffers[normalBufferView->buffer];
            }

            const tinygltf::Accessor* tangentAccessor = nullptr;
            const tinygltf::BufferView* tangentBufferView = nullptr;
            const tinygltf::Buffer* tangentBuffer = nullptr;

            if (loadTangent)
            {
                tangentAccessor = &model.accessors[primitive.attributes.at("TANGENT")];
                tangentBufferView = &model.bufferViews[tangentAccessor->bufferView];
                tangentBuffer = &model.buffers[tangentBufferView->buffer];
            }

            uint32_t baseVert = mesh_data_.vertices.size();

            // ------------------------
            // BUILD VERTEX LIST
            // ------------------------
            for (size_t i = 0; i < posAccessor.count; i++)
            {
                if (loadTangent)
                {
                    Vertex v;
                    const float* p = reinterpret_cast<const float*>(
                        &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.pos = glm::vec3(p[0], p[1], p[2]);

                    if (hasUV)
                    {
                        const float* uvp = reinterpret_cast<const float*>(
                            &uvBuffer->data[uvBufferView->byteOffset + uvAccessor->byteOffset + i * sizeof(glm::vec2)]
                            );
                        v.uv = glm::vec2(uvp[0], uvp[1]);
                    }

                    const float* np = reinterpret_cast<const float*>(
                        &normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.normal = glm::vec3(np[0], np[1], np[2]);

                    const float* tp = reinterpret_cast<const float*>(
                        &tangentBuffer->data[tangentBufferView->byteOffset + tangentAccessor->byteOffset + i * sizeof(glm::vec4)]
                        );
                    v.tangent = glm::vec4(tp[0], tp[1], tp[2], tp[3]);

                    mesh_data_.vertices.push_back(v);
                }
                else if (loadNormal)
                {
                    Vertex v;
                    const float* p = reinterpret_cast<const float*>(
                        &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.pos = glm::vec3(p[0], p[1], p[2]);

                    if (hasUV)
                    {
                        const float* uvp = reinterpret_cast<const float*>(
                            &uvBuffer->data[uvBufferView->byteOffset + uvAccessor->byteOffset + i * sizeof(glm::vec2)]
                            );
                        v.uv = glm::vec2(uvp[0], uvp[1]);
                    }

                    const float* np = reinterpret_cast<const float*>(
                        &normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.normal = glm::vec3(np[0], np[1], np[2]);

                    mesh_data_.vertices.push_back(v);
                }
                else
                {
                    Vertex v;
                    const float* p = reinterpret_cast<const float*>(
                        &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.pos = glm::vec3(p[0], p[1], p[2]);

                    if (hasUV)
                    {
                        const float* uvp = reinterpret_cast<const float*>(
                            &uvBuffer->data[uvBufferView->byteOffset + uvAccessor->byteOffset + i * sizeof(glm::vec2)]
                            );
                        v.uv = glm::vec2(uvp[0], uvp[1]);
                    }

                    mesh_data_.vertices.push_back(v);
                }
            }

            // ------------------------
            // BUILD INDICES
            // ------------------------
            const auto& idxAccessor = model.accessors[primitive.indices];
            const auto& idxView = model.bufferViews[idxAccessor.bufferView];
            const auto& idxBuffer = model.buffers[idxView.buffer];

            const unsigned char* idxData =
                &idxBuffer.data[idxView.byteOffset + idxAccessor.byteOffset];

            for (size_t i = 0; i < idxAccessor.count; i++)
            {
                uint32_t raw = 0;

                if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    raw = *reinterpret_cast<const uint16_t*>(idxData + i * sizeof(uint16_t));
                else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                    raw = *reinterpret_cast<const uint32_t*>(idxData + i * sizeof(uint32_t));
                else if (idxAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                    raw = *reinterpret_cast<const uint8_t*>(idxData + i * sizeof(uint8_t));

                mesh_data_.indices.push_back(baseVert + raw);
            }
            mesh_data_.indices_count = mesh_data_.indices.size();
        }
    }
}

void Model::ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta)
{   // 1. 컴포넌트를 직접 업데이트한다.

    // 기존 위치에 이동량을 더한다.
    position_ += translationDelta;

    // 기존 회전에 새로운 회전을 '앞에' 곱해준다.
    // (q_new * q_old 는 old 회전 후 new 회전을 적용하는 것과 같음)
    rotation_ = rotationDelta * rotation_;

    // 쿼터니언은 부동소수점 오차로 길이가 1이 아니게 될 수 있으므로,
    // 주기적으로 정규화해주는 것이 좋다.
    rotation_ = glm::normalize(rotation_);

    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale_);
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation_); // 쿼터니언 -> 회전 행렬
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

    // 2. SRT (Scale -> Rotate -> Translate) 순서로 조합하여 최종 월드 행렬을 계산한다.
    world_ = translationMatrix * rotationMatrix * scaleMatrix;
}