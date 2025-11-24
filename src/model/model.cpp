
#include "context.h"
#include "graphics_context.h"
#include "vertex.h"
#include "camera.h"
#include "geometry_generator.h"
#include "texture_manager.h"

#include "model.h"

Model::Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool movable)
{
    LoadModel(modelPath, vertexIncludeInfo, textureManager);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.vertices, mesh_data_.vertex_buffer, mesh_data_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.indices, mesh_data_.index_buffer, mesh_data_.index_buffer_memory);

    albedo_use_ = colorUse;
    movable_ = movable;

    ApplyTransform(initRotation, initPos);
}

Model::Model(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble)
{
    mesh_data_ = std::move(meshData);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.vertices, mesh_data_.vertex_buffer, mesh_data_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_data_.indices, mesh_data_.index_buffer, mesh_data_.index_buffer_memory);

    albedo_use_ = colorUse;
    movable_ = moveble;

    ApplyTransform(initRotation, initPos);
}

void Model::LoadModel(const std::string& modelPath, const vku::VertexIncludeInfo& vertexIncludeInfo, TextureManager& textureManager)
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
            bool hasUV = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();

            bool loadNormal = vertexIncludeInfo.normal && hasNormals;

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

            uint32_t baseVert = mesh_data_.vertices.size();

            for (size_t i = 0; i < posAccessor.count; i++)
            {
                Vertex v{};

                // pos
                const float* p = reinterpret_cast<const float*>(
                    &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * sizeof(glm::vec3)]
                    );
                v.pos = glm::vec3(p[0], p[1], p[2]);

                // uv
                if (hasUV)
                {
                    const float* uvp = reinterpret_cast<const float*>(
                        &uvBuffer->data[uvBufferView->byteOffset + uvAccessor->byteOffset + i * sizeof(glm::vec2)]
                        );
                    v.uv = glm::vec2(uvp[0], uvp[1]);
                }

                // normal
                if (loadNormal)
                {
                    const float* np = reinterpret_cast<const float*>(
                        &normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * sizeof(glm::vec3)]
                        );
                    v.normal = glm::vec3(np[0], np[1], np[2]);
                }

                v.tangent = glm::vec3(0.0f);

                mesh_data_.vertices.push_back(v);
            }

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

            //int materialIndex = primitive.material;
            //if (materialIndex >= 0 && materialIndex < static_cast<int>(model.materials.size()))
            //{
            //    const tinygltf::Material& mat = model.materials[materialIndex];

            //    auto getTexturePath = [&](int textureIndex) -> std::string
            //        {
            //            if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
            //                return {};

            //            const tinygltf::Texture& tex = model.textures[textureIndex];
            //            int imageIndex = tex.source;
            //            if (imageIndex < 0 || imageIndex >= static_cast<int>(model.images.size()))
            //                return {};

            //            const tinygltf::Image& img = model.images[imageIndex];
            //            if (img.uri.empty())
            //            {
            //                return {};
            //            }

            //            std::filesystem::path baseDir = std::filesystem::path(modelPath).parent_path();
            //            std::filesystem::path fullPath = baseDir / img.uri;
            //            return baseDir.string();
            //        };

            //    int baseColorTexIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
            //    int metallicRoughnessTexIdx = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
            //    int normalTexIndex = mat.normalTexture.index;
            //    int occlusionTexIndex = mat.occlusionTexture.index;
            //    int emissiveTexIndex = mat.emissiveTexture.index;

            //    std::string baseColorPath = getTexturePath(baseColorTexIndex);
            //    std::string metallicRoughPath = getTexturePath(metallicRoughnessTexIdx);
            //    std::string normalPath = getTexturePath(normalTexIndex);
            //    //std::string aoPath = getTexturePath(occlusionTexIndex);
            //    //std::string emissivePath = getTexturePath(emissiveTexIndex);

            //    texture_idx_.albedo = textureManager.CreateTexture(baseColorPath, "basecolor");
            //    texture_idx_.metallic = textureManager.CreateTexture(metallicRoughPath, "metallic");
            //    texture_idx_.normal = textureManager.CreateTexture(normalPath, "normal");
            //}

            if (vertexIncludeInfo.tangent) {
                GeometryGenerator::CalculateTangents(mesh_data_);
            }
        }
    }
}

void Model::ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta)
{
    position_ += translationDelta;
    rotation_ = rotationDelta * rotation_;
    rotation_ = glm::normalize(rotation_);

    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale_);
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation_);
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

    world_ = translationMatrix * rotationMatrix * scaleMatrix;
}