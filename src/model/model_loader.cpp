#include "texture.h"
#include "texture_manager.h"
#include "geometry_generator.h"

#include "model_loader.h"

ModelLoader::ModelLoader()
{

}

ModelLoader::~ModelLoader()
{

}

void ModelLoader::LoadModel(std::string modelPath, vku::VertexIncludeInfo vertexIncludeInfo, TextureManager& textureManager, float scale)
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
	ParseAnimations(model);

	bool hasSkin = !skin_.empty();

	if (!hasSkin) {
		int rootMeshNode = -1;
		for (int i = 0; i < (int)node_.size(); ++i) {
			if (node_[i].meshIndex >= 0 && node_[i].parent < 0) {
				rootMeshNode = i;
				break;
			}
		}

		if (rootMeshNode >= 0) {
			glm::mat4 M = node_[rootMeshNode].localMatrix; // or worldMatrix

			for (auto& v : mesh_.vertices) {
				glm::vec4 p = M * glm::vec4(v.pos, 1.0f);
				v.pos = glm::vec3(p);
			}
		}
	}

}

template<typename T>
std::vector<T> ModelLoader::ReadAccessor(const tinygltf::Model& model, int accessorIndex)
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

void ModelLoader::ParseMesh(const tinygltf::Model& model, vku::VertexIncludeInfo& vertexIncludeInfo, float scale)
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
						joints[i] = glm::uvec4(tmp[i]); // u8 �� u32
					}
				}
				else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
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


void ModelLoader::ParseNodes(const tinygltf::Model& model)
{
	for (int i = 0; i < (int)model.nodes.size(); ++i)
	{
		const tinygltf::Node& n = model.nodes[i];
		Node node;
		node.index = i;
		node.name = n.name;

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



void ModelLoader::ParseSkins(const tinygltf::Model& model)
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
		skin.skeletonRoot = s.skeleton;

		skin_.push_back(std::move(skin));
	}

}

void ModelLoader::ParseAnimations(const tinygltf::Model& model)
{
	animations_.clear();
	animations_.reserve(model.animations.size());

	for (size_t ai = 0; ai < model.animations.size(); ++ai)
	{
		const tinygltf::Animation& anim = model.animations[ai];
		AnimationClip clip;
		clip.name = anim.name.empty() ? ("Anim_" + std::to_string(ai)) : anim.name;

		for (const auto& channel : anim.channels)
		{
			if (channel.target_node < 0) continue;

			AnimChannel ch;
			ch.nodeIndex = channel.target_node;

			if (channel.target_path == "translation")
				ch.path = AnimPath::Translation;
			else if (channel.target_path == "rotation")
				ch.path = AnimPath::Rotation;
			else if (channel.target_path == "scale")
				ch.path = AnimPath::Scale;
			else
				continue;

			int samplerIndex = channel.sampler;
			if (samplerIndex < 0 || samplerIndex >= (int)anim.samplers.size())
				continue;

			const tinygltf::AnimationSampler& sampler = anim.samplers[samplerIndex];

			ch.times = ReadAccessor<float>(model, sampler.input);
			if (!ch.times.empty())
				clip.duration = std::max(clip.duration, ch.times.back());

			if (ch.path == AnimPath::Translation || ch.path == AnimPath::Scale)
			{
				auto vals = ReadAccessor<glm::vec3>(model, sampler.output);
				ch.values.resize(vals.size());
				for (size_t i = 0; i < vals.size(); ++i)
				{
					ch.values[i] = glm::vec4(vals[i], 0.0f);
				}
			}
			else if (ch.path == AnimPath::Rotation)
			{
				struct Vec4 { float x, y, z, w; };
				auto vals = ReadAccessor<Vec4>(model, sampler.output);
				ch.values.resize(vals.size());
				for (size_t i = 0; i < vals.size(); ++i)
				{
					ch.values[i] = glm::vec4(vals[i].x, vals[i].y, vals[i].z, vals[i].w);
				}
			}

			clip.channels.push_back(std::move(ch));
		}

		animations_.push_back(std::move(clip));
	}
}