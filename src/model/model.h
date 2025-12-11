#pragma once

class Context;
class PassManager;
struct Vertex;
struct Camera;
class TextureManager;

#include "model_data.h"
#include "vulkan_utils.h"

class Model
{
public:
	Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble, std::string name, float scale);
	Model(Mesh& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble, std::string name);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;
	~Model() = default;

	void ApplyTransform(glm::vec3 scaleDelta, glm::quat rotationDelta, glm::vec3 translationDelta);
	void ApplyAnimation(int clipIndex, float t);
	void UpdateSkinMatrices();
	void UpdateCollidersFromBones();

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
	float radius_ = 1.0f;
	bool checker_board_enable_ = false;
	bool movable_ = false;

	enum Type {
		NORMAL,
		SKINNED
	} type_;

	std::string name_;
	glm::vec4 albedo_{ 0.0f };

	Mesh mesh_;
	std::vector<Node> node_;
	std::vector<Skin> skin_;
	std::vector<AnimationClip> animations_;
	int   current_clip_ = 0;
	float current_time_ = 0.0f; // second

	std::vector<CapsuleColliderDef> collider_defs_;
	std::vector<CapsuleInstance> collider_instances_;

	struct TextureIdx {
		int albedo = -1;
		int metallic = -1;
		int normal = -1;
		int roughness = -1;
		int ao = -1;
		int height = -1;
		int emissive = -1;
	} texture_idx_;

	struct TextureEnable {
		bool albedo = 0;
		bool metallic = 0;
		bool normal = 0;
		bool roughness = 0;
		bool ao = 0;
		bool height = 0;
	} texture_enable_;

	struct Factor {
		float metallic = 0.0f;
		float roughness = 1.0f;
		float ao = 1.0f;
		float height = 0.0f;
		float coat = 0.0f;
		float coat_roughness = 0.0f;
		float fuzz = 0.0f;
		float fuzz_roughness = 0.0f;
	} factors_;

private:
	void LoadModel(const std::string& modelPath, vku::VertexIncludeInfo& vertexIncludeInfo, TextureManager& textureManager, float scale);
	template<typename T>
	std::vector<T> ReadAccessor(const tinygltf::Model& model, int accessorIndex);
	void ParseMesh(const tinygltf::Model& model, vku::VertexIncludeInfo& vertexIncludeInfo, float scale);
	void ParseNodes(const tinygltf::Model& model);
	void UpdateWorldTransforms(std::vector<Node>& nodes, int nodeIndex, const glm::mat4& parent);
	void ParseSkins(const tinygltf::Model& model);
	void ParseAnimations(const tinygltf::Model& model);
	void SetupColliders();
};