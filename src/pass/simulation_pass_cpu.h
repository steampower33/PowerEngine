#pragma once

class Context;
class Swapchain;
class Model;
class ModelManager;
class Texture;
class TextureManager;
class Ray;
class MouseInteractor;
class ParticleManager;

#include "sim_data.h"
#include "sim_ubo.h"

class SimulationPassCPU {

public:
	SimulationPassCPU(Context& context, ParticleManager& particleManager);
	SimulationPassCPU(const SimulationPassCPU& rhs) = delete;
	SimulationPassCPU(SimulationPassCPU&& rhs) = delete;
	SimulationPassCPU& operator=(const SimulationPassCPU& rhs) = delete;
	SimulationPassCPU& operator=(SimulationPassCPU&& rhs) = delete;
	~SimulationPassCPU() = default;

	Context& context_;
	std::vector<vk::raii::CommandBuffer> cmds_;

	void CopyDataToGPU(uint32_t currentFrmae);
	void ComputeSolve(ModelManager& modelManager);
	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);

	ParticleManager& particle_manager_;
	SimData datas_;
	uint32_t total_particles_;

	int id = -1;
	float dist2 = 1000.0f;
	float T = 1000.0f;

	struct PushConstant {
		struct MouseInteract {
			glm::vec3 ray_origin;
			uint32_t select_mode; // 0: none, 1: select, 2: drag
			glm::vec3 ray_dir;
			float radius = 0.1f;
		} mouse_interact;
		static_assert(sizeof(MouseInteract) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct ClothRender {
			uint32_t nx1;
			uint32_t ny1;
		} cloth_render;
		static_assert(sizeof(ClothRender) % 4 == 0, "push constant must be multiple of 4 bytes");

	} push_constants_;



private:
	struct CellKey
	{
		int x, y, z;
		bool operator==(const CellKey& o) const noexcept {
			return x == o.x && y == o.y && z == o.z;
		}
	};

	struct CellKeyHash
	{
		size_t operator()(const CellKey& k) const noexcept {
			size_t h = 1469598103934665603ull;
			auto mix = [&](int v) {
				h ^= std::hash<int>{}(v)+0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
				};
			mix(k.x);
			mix(k.y);
			mix(k.z);
			return h;
		}
	};

	using CellMap = std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash>;

	struct CollisionPair
	{
		uint32_t i;
		uint32_t j;
	};

	void BuildSpatialHash(float cellSize, CellMap& outCells);
	void BuildCollisionPairs(const CellMap& cells, float cellSize, std::vector<CollisionPair>& outPairs);
	void SolveSelfCollision(const std::vector<CollisionPair>& pairs, float thickness);

private:
	void CreateCommandBuffers();
	void CreateConstraintDatas();
};