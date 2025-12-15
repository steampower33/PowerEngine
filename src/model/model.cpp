
#include "context.h"
#include "pass_manager.h"
#include "vertex.h"
#include "camera.h"
#include "geometry_generator.h"
#include "texture_manager.h"
#include "model_loader.h"

#include "model.h"

Model::Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 initColor, float initRadius, bool moveble, std::string name, float scale, ShapeColliderType colliderType, ModelType modelType)
	: albedo_(initColor), movable_(moveble), name_(name), radius_(initRadius), shape_collision_type_(colliderType), model_type_(modelType)
{
	model_loader_ = std::make_unique<ModelLoader>();

	model_loader_->LoadModel(modelPath, vertexIncludeInfo, textureManager, scale);

	auto& mesh = model_loader_->mesh_;
	vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh.vertices, mesh.vertex_buffer, mesh.vertex_buffer_memory);
	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh.indices, mesh.index_buffer, mesh.index_buffer_memory);

	auto& node = model_loader_->node_;
	for (int i = 0; i < (int)node.size(); ++i) {
		if (node[i].parent < 0) {
			UpdateWorldTransforms(node, i, glm::mat4(1.0f));
		}
	}

	UpdateSkinMatrices();

	SetupCapsuleColliders();
	UpdateCapsuleCollidersFromBones();

	SetupShapeColliders();
	ApplyTransform(glm::vec3(1.0f), initRotation, initPos);
}

Model::Model(Mesh& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, glm::vec3 initPos, glm::quat initRotation, glm::vec4 initColor, float initRadius, bool moveble, std::string name, ShapeColliderType colliderType, ModelType modelType)
	: albedo_(initColor), movable_(moveble), name_(name), radius_(initRadius), shape_collision_type_(colliderType), model_type_(modelType)
{
	model_loader_ = std::make_unique<ModelLoader>();
	model_loader_->mesh_ = std::move(meshData);

	auto& mesh = model_loader_->mesh_;
	vku::CreateVertexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh.vertices, mesh.vertex_buffer, mesh.vertex_buffer_memory);
	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, mesh.indices, mesh.index_buffer, mesh.index_buffer_memory);

	SetupShapeColliders();
	ApplyTransform(glm::vec3(1.0f), initRotation, initPos);
}

void Model::UpdateShapeColliders()
{
	for (size_t i = 0; i < shape_colliders_.size(); ++i) {
		shape_colliders_[i].p0 = position_;
		shape_colliders_[i].do_collide = shape_collision_collide_;
	}
}

void Model::UpdateCapsuleCollidersFromBones()
{
	auto& node = model_loader_->node_;

	for (size_t i = 0; i < collider_defs_.size(); ++i) {
		const auto& def = collider_defs_[i];

		const glm::mat4& M0 = node[def.jointA].worldMatrix;
		const glm::mat4& M1 = node[def.jointB].worldMatrix;

		glm::vec3 p0 = glm::vec3(M0 * glm::vec4(0, 0, 0, 1));
		glm::vec3 p1 = glm::vec3(M1 * glm::vec4(0, 0, 0, 1));

		capsule_colliders_[i].p0 = p0;
		capsule_colliders_[i].kind = ShapeColliderType::CAPSULE;
		capsule_colliders_[i].p1 = p1;
		capsule_colliders_[i].radius = def.radius;
		capsule_colliders_[i].do_collide = capsule_collision_collide_;
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
	auto& skin = model_loader_->skin_;
	auto& node = model_loader_->node_;
	for (auto& skin : skin)
	{
		size_t n = skin.joints.size();
		skin.jointMatrices.resize(n);

		for (size_t i = 0; i < n; ++i)
		{
			int jointNodeIndex = skin.joints[i];
			const glm::mat4& M = node[jointNodeIndex].worldMatrix;
			const glm::mat4& B = skin.inverseBindMatrices[i];

			skin.jointMatrices[i] = M * B;
		}
	}
}


void Model::UpdateWorldTransforms(std::vector<Node>& nodes, int nodeIndex, const glm::mat4& parent)
{
	Node& n = nodes[nodeIndex];
	n.worldMatrix = parent * n.localMatrix;
	for (int c : n.children)
		UpdateWorldTransforms(nodes, c, n.worldMatrix);
}

void Model::ApplyAnimation(int clipIndex, float t)
{
	auto& node = model_loader_->node_;
	auto& animations = model_loader_->animations_;

	if (clipIndex < 0 || clipIndex >= (int)animations.size()) return;
	AnimationClip& clip = animations[clipIndex];
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

		Node& n = node[ch.nodeIndex];

		// In-Place or None
		if (ch.path == AnimPath::Translation) {
			//glm::vec3 a(v1.x, v1.y, v1.z);
			//glm::vec3 b(v2.x, v2.y, v2.z);
			//node.translation = glm::mix(a, b, f);
		}
		else if (ch.path == AnimPath::Scale) {
			glm::vec3 a(v1.x, v1.y, v1.z);
			glm::vec3 b(v2.x, v2.y, v2.z);
			n.scale = glm::mix(a, b, f);
		}
		else if (ch.path == AnimPath::Rotation) {
			glm::quat qa(v1.w, v1.x, v1.y, v1.z);
			glm::quat qb(v2.w, v2.x, v2.y, v2.z);
			n.rotation = glm::normalize(glm::slerp(qa, qb, f));
		}

		n.localMatrix =
			glm::translate(glm::mat4(1.0f), n.translation) *
			glm::mat4_cast(n.rotation) *
			glm::scale(glm::mat4(1.0f), n.scale);
	}

	for (int i = 0; i < (int)node.size(); ++i) {
		if (node[i].parent < 0)
			UpdateWorldTransforms(node, i, glm::mat4(1.0f));
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
	c.do_collide = shape_collision_collide_;

	shape_colliders_.push_back(c);
}

void Model::SetupCapsuleColliders()
{
	auto& node = model_loader_->node_;

	collider_defs_.clear();

	std::function<void(int)> dfs = [&](int idx)
		{
			//std::cout << idx << " : " << node_[idx].name << std::endl;

			for (int c : node[idx].children) {
				glm::vec3 p0 = glm::vec3(node[idx].worldMatrix * glm::vec4(0, 0, 0, 1));
				glm::vec3 p1 = glm::vec3(node[c].worldMatrix * glm::vec4(0, 0, 0, 1));

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

	for (int i = 0; i < (int)node.size(); ++i) {
		if (node[i].parent < 0) {
			for (int c : node[i].children) {
				if (node[c].name.find("Alpha") != std::string::npos)
					continue;

				dfs(c);
			}
		}
	}

	capsule_colliders_.resize(collider_defs_.size());
}
