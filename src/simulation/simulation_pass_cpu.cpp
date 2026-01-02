#include "context.h"
#include "swapchain.h"
#include "model.h"
#include "model_manager.h"
#include "texture.h"
#include "texture_manager.h"
#include "vulkan_utils.h"
#include "ray.h"
#include "mouse_interactor.h"
#include "particle_manager.h"

#include "simulation_pass_cpu.h"

SimulationPassCPU::SimulationPassCPU(Context& context, ParticleManager& particleManager)
	: context_(context), particle_manager_(particleManager)
{
	CreateCommandBuffers();
	CreateConstraintDatas();
}

void SimulationPassCPU::CopyDataToGPU(uint32_t currentFrmae)
{
	auto& pm = particle_manager_;

	auto& cmd = cmds_[currentFrmae];

	cmd.reset();
	cmd.begin({});

	vku::CopyStagingToSSBO(cmd, pm.ssbo_size_.position, pm.staging_mapped_.position, pm.positions_, pm.staging_.position, pm.ssbos_.position,
		vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead,
		vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

	cmd.end();
}

void SimulationPassCPU::UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize)
{
	if (mouseInteractor.is_left_button_down_event)
	{
		push_constants_.mouse_interact.select_mode = 1;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		push_constants_.mouse_interact.ray_origin = ray.origin;
		push_constants_.mouse_interact.ray_dir = ray.direction;

	}
	else if (!mouseInteractor.is_left_button_down_event && mouseInteractor.is_left_down)
	{
		push_constants_.mouse_interact.select_mode = 2;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		push_constants_.mouse_interact.ray_origin = ray.origin;
		push_constants_.mouse_interact.ray_dir = ray.direction;
	}
	else
		push_constants_.mouse_interact.select_mode = 0;
}

void SimulationPassCPU::ComputeSolve(ModelManager& modelManager)
{
	auto& pm = particle_manager_;
	auto& d = datas_;
	auto& cloth = pm.clothes_[0];

	const uint32_t N = total_particles_;
	if (N == 0) return;

	const float frame_dt = 1.0f / d.frame_dt;
	const int   substeps = std::max(1, d.substeps);
	const float dt = frame_dt / static_cast<float>(substeps);

	const int iterations = std::max(1, d.iterations);

	const glm::vec3 gravity(0.0f, -9.81f, 0.0f);
	const float velocity_damping = 0.01f;

	auto& pc = push_constants_.mouse_interact;

	auto mouse_select = [&](int i, glm::vec3& pos) {
		if (pc.select_mode == 0)
		{
			id = -1;
			dist2 = 1000.0f;
			T = 1000.0f;
		}

		if (pc.select_mode != 1)
			return;

		glm::vec3 o = pc.ray_origin;
		glm::vec3 dir = pc.ray_dir;

		glm::vec3 v = pos - o;
		float t = std::max(dot(v, dir), 0.0f);
		glm::vec3 c = o + dir * t;
		float d2 = dot(pos - c, pos - c);

		float r = pc.radius;
		if (d2 > r * r) return;

		if (dist2 > d2)
		{
			id = i;
			dist2 = d2;
			T = t;
			//std::cout << "[Select] id : " << id << ", dist2 : " << dist2 << ", T : " << T << std::endl;
		}

		};

	auto mouse_drag = [&](int i, glm::vec3& xpi, glm::vec3& vi) {
		if (pc.select_mode != 2)
			return;

		if (dist2 >= 999.9f)
			return;

		if (i != id)
			return;

		glm::vec3 o = pc.ray_origin;
		glm::vec3 dir = pc.ray_dir;

		float t = T;
		glm::vec3 c = o + t * dir;

		xpi = c;
		vi = glm::vec3(0.0f);

		//std::cout << "[Drag] id : " << id << ", dist2 : " << dist2 << ", T : " << T << std::endl;
		};

	CellMap cells;
	std::vector<CollisionPair> collisionPairs;

	const float baseSpacing = cloth.spacing;
	const float cellSize = baseSpacing * 1.0f;   // broadphase cell size
	const float thickness = baseSpacing * 0.8f;   // Real minimum distance

	for (int step = 0; step < substeps; ++step)
	{
		for (uint32_t i = 0; i < N; ++i)
		{
			float w = pm.inverse_masses_[i];
			glm::vec3 x = glm::vec3(pm.positions_[i]);
			glm::vec3 v = glm::vec3(pm.velocities_[i]);

			if (w == 0.0f) {
				pm.pred_positions_[i] = pm.positions_[i];
				continue;
			}

			mouse_select(i, x);

			v += gravity * dt;

			glm::vec3 xpi = x + v * dt;

			mouse_drag(i, xpi, v);

			pm.pred_positions_[i] = glm::vec4(xpi, 0.0f);
			pm.velocities_[i] = glm::vec4(v, 0.0f);
		}

		for (auto& e : d.edges) {
			e.lambda = 0.0f;
		}

		BuildSpatialHash(cellSize, cells);
		BuildCollisionPairs(cells, cellSize, collisionPairs);

		for (int iter = 0; iter < iterations; ++iter)
		{
			const float alpha_stretch = d.compliance.stretch / (dt * dt);

			for (auto& e : d.edges)
			{
				uint32_t i = e.i;
				uint32_t j = e.j;

				float wi = pm.inverse_masses_[i];
				float wj = pm.inverse_masses_[j];
				if (wi + wj <= 0.0f)
					continue;

				glm::vec3 xi = glm::vec3(pm.pred_positions_[i]);
				glm::vec3 xj = glm::vec3(pm.pred_positions_[j]);

				glm::vec3 n = xi - xj;
				float L = glm::length(n);
				if (L < 1e-8f)
					continue;

				n /= L;

				float C = L - e.rest;

				float denom = wi + wj + alpha_stretch;
				float dl = -(C + alpha_stretch * e.lambda) / denom;
				e.lambda += dl;

				glm::vec3 corr = dl * n;

				if (wi > 0.0f)
					pm.pred_positions_[i] += glm::vec4(wi * corr, 0.0f);
				if (wj > 0.0f)
					pm.pred_positions_[j] -= glm::vec4(wj * corr, 0.0f);
			}

			for (uint32_t i = 0; i < N; ++i)
			{
				float w = pm.inverse_masses_[i];
				if (w == 0.0f) continue;

				glm::vec3 p = glm::vec3(pm.pred_positions_[i]);
				glm::vec3 sphereCenter = modelManager.models_[0]->position_;
				float sphereRadius = modelManager.models_[0]->radius_;
				glm::vec3 dvec = p - sphereCenter;
				float dist = glm::length(dvec);

				if (dist < modelManager.models_[0]->radius_)
				{
					glm::vec3 n = (dist > 1e-8f) ? (dvec / dist) : glm::vec3(0, 1, 0);
					p = sphereCenter + n * sphereRadius;
					pm.pred_positions_[i] = glm::vec4(p, 1.0f);
				}

				if (p.y < 0.0f)
				{
					pm.pred_positions_[i].y = 0.0f;
				}
			}

			SolveSelfCollision(collisionPairs, thickness);
				
			// shear / bend / area
		}

		for (uint32_t i = 0; i < N; ++i)
		{
			float w = pm.inverse_masses_[i];
			glm::vec3 x_old = glm::vec3(pm.positions_[i]);
			glm::vec3 x_new = glm::vec3(pm.pred_positions_[i]);

			if (w == 0.0f) {
				pm.positions_[i] = glm::vec4(x_old, 1.0f);
				pm.velocities_[i] = glm::vec4(0.0f);
				continue;
			}

			glm::vec3 v = (x_new - x_old) / dt;
			v *= (1.0f - velocity_damping);

			pm.positions_[i] = glm::vec4(x_new, 1.0f);
			pm.velocities_[i] = glm::vec4(v, 0.0f);
		}
	}
}

void SimulationPassCPU::BuildSpatialHash(float cellSize, CellMap& outCells)
{
	outCells.clear();

	auto& d = datas_;
	auto& pm = particle_manager_;

	const uint32_t N = total_particles_;
	if (N == 0) return;

	const float invCell = 1.0f / cellSize;

	for (uint32_t i = 0; i < N; ++i)
	{
		glm::vec3 p = glm::vec3(pm.pred_positions_[i]);

		int ix = static_cast<int>(std::floor(p.x * invCell));
		int iy = static_cast<int>(std::floor(p.y * invCell));
		int iz = static_cast<int>(std::floor(p.z * invCell));

		CellKey key{ ix, iy, iz };
		outCells[key].push_back(i);
	}
}

void SimulationPassCPU::BuildCollisionPairs(const CellMap& cells, float cellSize, std::vector<CollisionPair>& outPairs)
{
	outPairs.clear();

	auto& d = datas_;
	auto& pm = particle_manager_;

	const uint32_t N = total_particles_;
	if (N == 0) return;

	const float invCell = 1.0f / cellSize;

	for (uint32_t i = 0; i < N; ++i)
	{
		glm::vec3 p = glm::vec3(pm.pred_positions_[i]);
		int ix = static_cast<int>(std::floor(p.x * invCell));
		int iy = static_cast<int>(std::floor(p.y * invCell));
		int iz = static_cast<int>(std::floor(p.z * invCell));

		for (int dz = -1; dz <= 1; ++dz)
			for (int dy = -1; dy <= 1; ++dy)
				for (int dx = -1; dx <= 1; ++dx)
				{
					CellKey key{ ix + dx, iy + dy, iz + dz };
					auto it = cells.find(key);
					if (it == cells.end()) continue;

					const auto& bucket = it->second;
					for (uint32_t j : bucket)
					{
						if (j <= i) continue;

						outPairs.push_back({ i, j });
					}
				}
	}
}

void SimulationPassCPU::SolveSelfCollision(const std::vector<CollisionPair>& pairs, float thickness)
{
	auto& d = datas_;
	auto& pm = particle_manager_;

	const float thickness2 = thickness * thickness;

	for (const auto& pair : pairs)
	{
		uint32_t i = pair.i;
		uint32_t j = pair.j;

		float wi = pm.inverse_masses_[i];
		float wj = pm.inverse_masses_[j];
		if (wi + wj <= 0.0f) continue;

		glm::vec3 xi = glm::vec3(pm.pred_positions_[i]);
		glm::vec3 xj = glm::vec3(pm.pred_positions_[j]);

		glm::vec3 dvec = xi - xj;
		float dist2 = glm::dot(dvec, dvec);
		if (dist2 < 1e-12f) continue;

		if (dist2 >= thickness2) continue;

		float dist = std::sqrt(dist2);
		glm::vec3 n = dvec / dist;

		float C = thickness - dist;
		if (C <= 0.0f) continue;

		float wSum = wi + wj;
		glm::vec3 corr = (C / wSum) * n;

		if (wi > 0.0f)
			pm.pred_positions_[i] += glm::vec4(wi * corr, 0.0f);
		if (wj > 0.0f)
			pm.pred_positions_[j] -= glm::vec4(wj * corr, 0.0f);
	}
}

void SimulationPassCPU::CreateCommandBuffers()
{
	cmds_.clear();
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.commandPool = *context_.command_pool_;
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
	cmds_ = vk::raii::CommandBuffers(context_.device_, allocInfo);
}


void SimulationPassCPU::CreateConstraintDatas()
{
	auto& pm = particle_manager_;
	auto& d = datas_;
	auto& cloth = pm.clothes_[0];

	total_particles_ = cloth.num_particle;
	uint32_t N = total_particles_;

	//auto CreateColoringPass = [&](Cloth cloth)
	//	{
	//		uint32_t nx1 = cloth.nx + 1;
	//		uint32_t ny1 = cloth.ny + 1;

	//		auto vid = [&](int x, int y) {
	//			return cloth.offset_particle + uint32_t(y * nx1 + x);
	//			};

	//		// Set stretch edges coloring
	//		for (int x = 0; x < nx1; ++x)
	//			for (int y = 0; y + 1 < ny1; y += 2)
	//				datas_.passes[0].push_back({ vid(x,y), vid(x,y + 1) });

	//		for (int x = 0; x < nx1; ++x)
	//			for (int y = 1; y + 1 < ny1; y += 2)
	//				datas_.passes[1].push_back({ vid(x,y), vid(x,y + 1) });

	//		for (int y = 0; y < ny1; ++y)
	//			for (int x = 0; x + 1 < nx1; x += 2)
	//				datas_.passes[2].push_back({ vid(x,y), vid(x + 1,y) });

	//		for (int y = 0; y < ny1; ++y)
	//			for (int x = 1; x + 1 < nx1; x += 2)
	//				datas_.passes[3].push_back({ vid(x,y), vid(x + 1,y) });
	//	};

	//CreateColoringPass(cloth);

	//// Set Edges using coloring
	//datas_.pass_offsets[0] = 0;

	//for (int p = 0; p < datas_.pass_offsets.size() - 1; ++p) {
	//	for (auto [i, j] : datas_.passes[p]) {
	//		glm::vec3 pi = glm::vec3(pm.positions_[i]);
	//		glm::vec3 pj = glm::vec3(pm.positions_[j]);
	//		float rest = glm::length(pj - pi);

	//		datas_.edges.push_back({ i, j, rest, 0.0f });
	//	}
	//	datas_.pass_offsets[p + 1] = static_cast<uint32_t>(datas_.edges.size());
	//}

	datas_.num_edges = static_cast<uint32_t>(datas_.edges.size());
	//if (datas_.num_edges != ((nx1 - 1) * ny1) + (nx1 * (ny1 - 1)))
	//{
	//	std::cout << datas_.num_edges << " " << ((nx1 - 1) * ny1) + (nx1 * (ny1 - 1)) << std::endl;
	//	throw std::runtime_error("edge size is not right");
	//}

	//// Set shears
	//const size_t numTris = pm.indices.size() / 3;
	//datas_.shears.reserve(numTris);

	//for (size_t t = 0; t < numTris; ++t)
	//{
	//	uint32_t i0 = pm.indices[3 * t + 0];
	//	uint32_t i1 = pm.indices[3 * t + 1];
	//	uint32_t i2 = pm.indices[3 * t + 2];

	//	const glm::vec3& x0 = pm.positions[i0];
	//	const glm::vec3& x1 = pm.positions[i1];
	//	const glm::vec3& x2 = pm.positions[i2];

	//	glm::vec3 e1 = x1 - x0;
	//	glm::vec3 e2 = x2 - x0;

	//	float restDot = glm::dot(e1, e2);

	//	SimData::Shear c;
	//	c.i0 = i0;
	//	c.i1 = i1;
	//	c.i2 = i2;
	//	c.rest_dot = restDot;
	//	c.lambda = 0.0f;

	//	datas_.shears.push_back(c);
	//}
	//datas_.num_shears = static_cast<uint32_t>(datas_.shears.size());

	//datas_.BuildBendConstraints(pm.positions, pm.indices);
	//datas_.num_bends = static_cast<uint32_t>(datas_.bends.size());

	//datas_.BuildAreaConstraints(pm.positions, pm.indices);
	//datas_.num_areas = static_cast<uint32_t>(datas_.areas.size());

}