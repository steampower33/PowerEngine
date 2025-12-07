
#include "context.h"
#include "pass_manager.h"
#include "vertex.h"
#include "camera.h"
#include "geometry_generator.h"
#include "texture_manager.h"

#include "model.h"

Model::Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool movable, std::string name, float scale)
{
    LoadModel(modelPath, vertexIncludeInfo, textureManager, scale);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.vertices, mesh_.vertex_buffer, mesh_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.indices, mesh_.index_buffer, mesh_.index_buffer_memory);

    albedo_ = colorUse;
    movable_ = movable;
    name_ = name;

    ApplyTransform(glm::vec3(1.0f), initRotation, initPos);
}

Model::Model(Mesh& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble, std::string name)
{
    mesh_ = std::move(meshData);

    vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.vertices, mesh_.vertex_buffer, mesh_.vertex_buffer_memory);
    vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.indices, mesh_.index_buffer, mesh_.index_buffer_memory);

    albedo_ = colorUse;
    movable_ = moveble;
    name_ = name;

    ApplyTransform(glm::vec3(1.0f), initRotation, initPos);
}

void Model::LoadModel(const std::string& modelPath, vku::VertexIncludeInfo& vertexIncludeInfo, TextureManager& textureManager, float scale)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ret;
    if (modelPath.find(".gltf") != std::string::npos)
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelPath);
    else if (modelPath.find(".glb") != std::string::npos)
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, modelPath);
    if (!warn.empty()) std::cout << "Warn: " << warn << std::endl;
    if (!err.empty())  std::cerr << "Err: " << err << std::endl;
    if (!ret) throw std::runtime_error("Failed to load glTF model");

    ParseMesh(model, vertexIncludeInfo, scale);
    ParseNodes(model);
    ParseSkins(model);

    for (int i = 0; i < (int)node_.size(); ++i) {
        if (node_[i].parent < 0) {
            UpdateWorldTransforms(node_, i, glm::mat4(1.0f));
        }
    }
}

void Model::ApplyTransform(glm::vec3 scaleDelta, glm::quat rotationDelta, glm::vec3 translationDelta)
{
    position_ += translationDelta;
    rotation_ = rotationDelta * rotation_;
    rotation_ = glm::normalize(rotation_);

    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scaleDelta);
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation_);
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

    world_ = translationMatrix * rotationMatrix * scaleMatrix;
}

void Model::ParseMesh(const tinygltf::Model& model, vku::VertexIncludeInfo& vertexIncludeInfo, float scale)
{
    mesh_.vertices.clear();
    mesh_.indices.clear();

    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            const auto& attrs = primitive.attributes;

            auto itPos = attrs.find("POSITION");
            assert(itPos != attrs.end());
            auto positions = ReadAccessor<glm::vec3>(model, itPos->second);

            std::vector<glm::vec2> uvs;
            if (auto it = attrs.find("TEXCOORD_0"); it != attrs.end())
                uvs = ReadAccessor<glm::vec2>(model, it->second);

            std::vector<glm::vec3> normals;
            if (auto it = attrs.find("NORMAL"); it != attrs.end())
                normals = ReadAccessor<glm::vec3>(model, it->second);

            // add later check UNSIGNED_BYTE or UNSIGNED_SHORT type 
            std::vector<glm::uvec4> joints;
            if (auto it = attrs.find("JOINTS_0"); it != attrs.end()) {
                const auto& acc = model.accessors[it->second];

                if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    auto tmp = ReadAccessor<glm::u8vec4>(model, it->second);
                    joints.resize(tmp.size());
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        joints[i] = glm::uvec4(tmp[i]); // u8 → u32
                    }
                }
                else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    // u16vec4 직접 만들어서 u32로 올리기
                    struct U16Vec4 { uint16_t x, y, z, w; };
                    auto tmp = ReadAccessor<U16Vec4>(model, it->second);
                    joints.resize(tmp.size());
                    for (size_t i = 0; i < tmp.size(); ++i) {
                        joints[i] = glm::uvec4(
                            tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w
                        );
                    }
                }
                else {
                    assert(false && "Unsupported JOINTS_0 componentType");
                }
            }
            std::vector<glm::vec4> weights;
            if (auto it = attrs.find("WEIGHTS_0"); it != attrs.end())
                weights = ReadAccessor<glm::vec4>(model, it->second);

            size_t vcount = positions.size();
            uint32_t baseVertex = static_cast<uint32_t>(mesh_.vertices.size());

            for (size_t i = 0; i < vcount; ++i)
            {
                Vertex v{};
                v.pos = positions[i] * scale;
                v.normal = normals.empty() ? glm::vec3(0, 1, 0) : normals[i];
                v.uv = uvs.empty() ? glm::vec2(0) : uvs[i];
                v.tangent = glm::vec3(0.0f);

                if (!joints.empty()) {
                    v.joints = glm::uvec4(joints[i]);
                }
                if (!weights.empty()) {
                    glm::vec4 w = weights[i];
                    float sum = w.x + w.y + w.z + w.w;
                    if (sum > 0.0f) w /= sum;
                    v.weights = w;
                }
                else {
                    v.weights = glm::vec4(1, 0, 0, 0);
                }

                mesh_.vertices.push_back(v);
            }

            if (primitive.indices >= 0)
            {
                const tinygltf::Accessor& acc = model.accessors[primitive.indices];
                const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
                const tinygltf::Buffer& buf = model.buffers[bv.buffer];

                const uint8_t* dataPtr = buf.data.data() + bv.byteOffset + acc.byteOffset;

                for (size_t i = 0; i < acc.count; ++i)
                {
                    uint32_t idx = 0;
                    switch (acc.componentType)
                    {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        idx = ((const uint16_t*)dataPtr)[i];
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                        idx = ((const uint32_t*)dataPtr)[i];
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        idx = ((const uint8_t*)dataPtr)[i];
                        break;
                    default:
                        assert(false && "Unsupported index type");
                    }

                    mesh_.indices.push_back(baseVertex + idx);
                }
            }
            mesh_.indices_count = mesh_.indices.size();
        }
    }
    if (vertexIncludeInfo.tangent) {
        GeometryGenerator::CalculateTangents(mesh_);
    }
}

template<typename T>
std::vector<T> Model::ReadAccessor(const tinygltf::Model& model, int accessorIndex)
{
    const tinygltf::Accessor& acc = model.accessors[accessorIndex];
    const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
    const tinygltf::Buffer& buf = model.buffers[bv.buffer];

    const uint8_t* dataPtr = buf.data.data() + bv.byteOffset + acc.byteOffset;
    size_t stride = acc.ByteStride(bv);
    if (stride == 0) stride = sizeof(T);

    std::vector<T> out(acc.count);
    for (size_t i = 0; i < acc.count; ++i)
    {
        memcpy(&out[i], dataPtr + i * stride, sizeof(T));
    }
    return out;
}

void Model::ParseNodes(const tinygltf::Model& model)
{
    for (int i = 0; i < (int)model.nodes.size(); ++i)
    {
        const tinygltf::Node& n = model.nodes[i];
        Node node;
        node.index = i;

        // children
        node.children.assign(n.children.begin(), n.children.end());

        // TRS or matrix
        auto trans = n.translation.data();
        if (!n.translation.empty())
            node.translation = glm::make_vec3(n.translation.data());
        if (!n.rotation.empty())
            node.rotation = glm::quat(
                (float)n.rotation[3],   // w
                (float)n.rotation[0],
                (float)n.rotation[1],
                (float)n.rotation[2]
            );
        if (!n.scale.empty())
        {
            auto scale = n.scale.data();
            node.scale = glm::make_vec3(n.scale.data());
        }

        if (!n.matrix.empty()) {
            node.localMatrix = glm::make_mat4x4(n.matrix.data());
        }
        else {
            node.localMatrix =
                glm::translate(glm::mat4(1.0f), node.translation) *
                glm::mat4_cast(node.rotation) *
                glm::scale(glm::mat4(1.0f), node.scale);
        }

        node.meshIndex = n.mesh;
        node.skinIndex = n.skin;

        node_.push_back(node);
    }

    for (int i = 0; i < (int)node_.size(); ++i)
    {
        for (int child : node_[i].children)
            node_[child].parent = i;
    }
}

void Model::UpdateWorldTransforms(std::vector<Node>& nodes, int nodeIndex, const glm::mat4& parent)
{
    Node& n = nodes[nodeIndex];
    n.worldMatrix = parent * n.localMatrix;
    for (int c : n.children)
        UpdateWorldTransforms(nodes, c, n.worldMatrix);
}

void Model::ParseSkins(const tinygltf::Model& model)
{
    skin_.reserve(model.skins.size());

    for (const auto& s : model.skins)
    {
        Skin skin;
        skin.name = s.name;
        skin.joints.assign(s.joints.begin(), s.joints.end());

        // inverseBindMatrices
        if (s.inverseBindMatrices >= 0)
        {
            const tinygltf::Accessor& acc = model.accessors[s.inverseBindMatrices];
            const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
            const tinygltf::Buffer& buf = model.buffers[bv.buffer];

            const uint8_t* dataPtr = buf.data.data() + bv.byteOffset + acc.byteOffset;
            size_t count = acc.count;
            skin.inverseBindMatrices.resize(count);

            for (size_t i = 0; i < count; ++i)
            {
                memcpy(glm::value_ptr(skin.inverseBindMatrices[i]),
                    dataPtr + i * sizeof(glm::mat4),
                    sizeof(glm::mat4));
            }
        }

        // skeleton root (optional)
        skin.skeletonRoot = s.skeleton; // -1이면 직접 찾거나 무시

        skin_.push_back(std::move(skin));
    }

}
