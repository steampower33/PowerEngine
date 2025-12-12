
#include "context.h"
#include "pass_manager.h"
#include "vertex.h"
#include "camera.h"
#include "geometry_generator.h"
#include "texture_manager.h"

#include "model.h"

Model::Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 initColor, float initRadius, bool moveble, std::string name, float scale, ShapeColliderType colliderType, ModelType modelType)
	: albedo_(initColor), movable_(moveble), name_(name), radius_(initRadius), shape_collision_type_(colliderType), model_type_(modelType)
{
	LoadModel(modelPath, vertexIncludeInfo, textureManager, scale);

	vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.vertices, mesh_.vertex_buffer, mesh_.vertex_buffer_memory);
	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.indices, mesh_.index_buffer, mesh_.index_buffer_memory);

	SetupShapeColliders();
	ApplyTransform(glm::vec3(1.0f), initRotation, initPos);
}

Model::Model(Mesh& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, glm::vec3 initPos, glm::quat initRotation, glm::vec4 initColor, float initRadius, bool moveble, std::string name, ShapeColliderType colliderType, ModelType modelType)
	: albedo_(initColor), movable_(moveble), name_(name), radius_(initRadius), shape_collision_type_(colliderType), model_type_(modelType)
{
	mesh_ = std::move(meshData);

	vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.vertices, mesh_.vertex_buffer, mesh_.vertex_buffer_memory);
	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh_.indices, mesh_.index_buffer, mesh_.index_buffer_memory);

	SetupShapeColliders();
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
	ParseAnimations(model);

	for (int i = 0; i < (int)node_.size(); ++i) {
		if (node_[i].parent < 0) {
			UpdateWorldTransforms(node_, i, glm::mat4(1.0f));
		}
	}

	UpdateSkinMatrices();

	SetupCapsuleColliders();
	UpdateCapsuleCollidersFromBones();
}

void Model::UpdateShapeColliders()
{
	for (size_t i = 0; i < shape_colliders_.size(); ++i) {
		shape_colliders_[i].p0 = position_;
	}
}

void Model::UpdateCapsuleCollidersFromBones()
{

	for (size_t i = 0; i < collider_defs_.size(); ++i) {
		const auto& def = collider_defs_[i];

		const glm::mat4& M0 = node_[def.jointA].worldMatrix;
		const glm::mat4& M1 = node_[def.jointB].worldMatrix;

		glm::vec3 p0 = glm::vec3(M0 * glm::vec4(0, 0, 0, 1));
		glm::vec3 p1 = glm::vec3(M1 * glm::vec4(0, 0, 0, 1));

		capsule_colliders_[i].p0 = p0;
		capsule_colliders_[i].kind = ShapeColliderType::CAPSULE;
		capsule_colliders_[i].p1 = p1;
		capsule_colliders_[i].radius = def.radius;
	}

	capsule_collision_update_ = true;
}

void Model::ApplyTransform(glm::vec3 scaleDelta, glm::quat rotationDelta, glm::vec3 translationDelta)
{
	shape_collision_update_ = true;

	position_ += translationDelta;
	rotation_ = rotationDelta * rotation_;
	rotation_ = glm::normalize(rotation_);

	glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scaleDelta);
	glm::mat4 rotationMatrix = glm::mat4_cast(rotation_);
	glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), position_);

	world_ = translationMatrix * rotationMatrix * scaleMatrix;

	UpdateShapeColliders();

}

void Model::UpdateSkinMatrices()
{
	for (auto& skin : skin_)
	{
		size_t n = skin.joints.size();
		skin.jointMatrices.resize(n);

		for (size_t i = 0; i < n; ++i)
		{
			int jointNodeIndex = skin.joints[i];
			const glm::mat4& M = node_[jointNodeIndex].worldMatrix;
			const glm::mat4& B = skin.inverseBindMatrices[i];

			skin.jointMatrices[i] = M * B;
		}
	}
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
						joints[i] = glm::uvec4(tmp[i]); // u8 ¡æ u32
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
		skin.skeletonRoot = s.skeleton;

		skin_.push_back(std::move(skin));
	}

}

void Model::ParseAnimations(const tinygltf::Model& model)
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

void Model::ApplyAnimation(int clipIndex, float t)
{
	if (clipIndex < 0 || clipIndex >= (int)animations_.size()) return;
	AnimationClip& clip = animations_[clipIndex];
	if (clip.duration <= 0.0f) return;

	float localTime = fmod(t, clip.duration);

	for (auto& ch : clip.channels)
	{
		if (ch.times.empty() || ch.values.size() != ch.times.size()) continue;

		int k1 = 0;
		while (k1 < (int)ch.times.size() - 1 && localTime > ch.times[k1 + 1])
			++k1;
		int k2 = std::min(k1 + 1, (int)ch.times.size() - 1);

		float t1 = ch.times[k1];
		float t2 = ch.times[k2];
		float f = (t2 > t1) ? ((localTime - t1) / (t2 - t1)) : 0.0f;
		f = glm::clamp(f, 0.0f, 1.0f);

		const glm::vec4 v1 = ch.values[k1];
		const glm::vec4 v2 = ch.values[k2];

		Node& node = node_[ch.nodeIndex];

		// In-Place or None
		if (ch.path == AnimPath::Translation) {
			//glm::vec3 a(v1.x, v1.y, v1.z);
			//glm::vec3 b(v2.x, v2.y, v2.z);
			//node.translation = glm::mix(a, b, f);
		}
		else if (ch.path == AnimPath::Scale) {
			glm::vec3 a(v1.x, v1.y, v1.z);
			glm::vec3 b(v2.x, v2.y, v2.z);
			node.scale = glm::mix(a, b, f);
		}
		else if (ch.path == AnimPath::Rotation) {
			glm::quat qa(v1.w, v1.x, v1.y, v1.z);
			glm::quat qb(v2.w, v2.x, v2.y, v2.z);
			node.rotation = glm::normalize(glm::slerp(qa, qb, f));
		}

		node.localMatrix =
			glm::translate(glm::mat4(1.0f), node.translation) *
			glm::mat4_cast(node.rotation) *
			glm::scale(glm::mat4(1.0f), node.scale);
	}

	for (int i = 0; i < (int)node_.size(); ++i) {
		if (node_[i].parent < 0)
			UpdateWorldTransforms(node_, i, glm::mat4(1.0f));
	}
	UpdateSkinMatrices();
}

void Model::SetupShapeColliders()
{
	Collider c;
	c.p0 = position_;
	c.kind = shape_collision_type_;
	c.p1 = glm::vec3(0.0f, 1.0f, 0.0f); // simple
	c.radius = radius_;

	shape_colliders_.push_back(c);
}

void Model::SetupCapsuleColliders()
{
	collider_defs_.clear();

	std::function<void(int)> dfs = [&](int idx)
		{
			//std::cout << idx << " : " << node_[idx].name << std::endl;

			for (int c : node_[idx].children) {
				glm::vec3 p0 = glm::vec3(node_[idx].worldMatrix * glm::vec4(0, 0, 0, 1));
				glm::vec3 p1 = glm::vec3(node_[c].worldMatrix * glm::vec4(0, 0, 0, 1));

				glm::vec3 seg = p1 - p0;
				float len = glm::length(seg);
				if (len < 1e-4f) {
					collider_defs_.push_back({ idx, c, 0.01f });
				}
				else {
					float radius = len * 0.25f;

					const float minR = 0.015f;
					const float maxR = 0.15f;
					radius = glm::clamp(radius, minR, maxR);

					collider_defs_.push_back({ idx, c, radius });
				}

				dfs(c);
			}
		};

	for (int i = 0; i < (int)node_.size(); ++i) {
		if (node_[i].parent < 0) {
			for (int c : node_[i].children) {
				if (node_[c].name.find("Alpha") != std::string::npos)
					continue;

				dfs(c);
			}
		}
	}

	capsule_colliders_.resize(collider_defs_.size());
}
