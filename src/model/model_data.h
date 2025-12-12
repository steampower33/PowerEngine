#pragma once

#include "vertex.h"

struct Mesh
{
	std::vector<Vertex> vertices;
	vk::raii::Buffer vertex_buffer{ nullptr };
	vk::raii::DeviceMemory vertex_buffer_memory{ nullptr };

	std::vector<uint32_t> indices; 
	vk::raii::Buffer index_buffer{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory{ nullptr };
	uint32_t indices_count;
};

struct Node
{
	int index = -1;
	int parent = -1;

	std::vector<int> children;
	std::string name;

	glm::vec3 translation = glm::vec3(0.0f);
	glm::quat rotation = glm::quat(1, 0, 0, 0);   // identity
	glm::vec3 scale = glm::vec3(1.0f);

	glm::mat4 localMatrix = glm::mat4(1.0f);
	glm::mat4 worldMatrix = glm::mat4(1.0f);

	int meshIndex = -1;
	int skinIndex = -1;
};

struct Skin
{
	std::string name;
	std::vector<int> joints;
	std::vector<glm::mat4> inverseBindMatrices;
	int skeletonRoot = -1;
	std::vector<glm::mat4> jointMatrices;
};

enum class AnimPath {
	Translation,
	Rotation,
	Scale
};

struct AnimChannel {
	int nodeIndex;
	AnimPath path;                      // T / R / S
	std::vector<float> times;           // keyframe times
	std::vector<glm::vec4> values;
};

struct AnimationClip {
	std::string name;
	float duration = 0.0f;              // Total length
	std::vector<AnimChannel> channels;
};

struct CapsuleColliderDef {
	int jointA;   // start bone index
	int jointB;   // end bone index
	float radius;
};

struct Collider {
	glm::vec3 p0;
	int kind; // kind: 0 = sphere, 1 = plane, 2 = capsule
	glm::vec3 p1;
	float radius;
};
static_assert(sizeof(Collider) == 32, "Collider must be 32 bytes");