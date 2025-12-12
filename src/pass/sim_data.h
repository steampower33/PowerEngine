#pragma once

#include "model_data.h"

struct SimData {

	struct Compliance {
		float stretch = 1e-6f;
		float softbody_stretch = 1e-6f;
		float softbody_volume = 1e-6f;
		float shear = 1e-6f;
		float bend = 1.0f;
		float area = 1.0f;
		float self_collision = 1e-9f;
	} compliance;

	struct Beta {
		float stretch = 100.0f;
	} beta;

	uint32_t num_edges = 0;
	uint32_t num_shears = 0;
	uint32_t num_bends = 0;
	uint32_t num_areas = 0;
	uint32_t num_volumes = 0;
	uint32_t num_colliders = 0;

	float frame_dt = 60.0f;
	int substeps = 10;
	int iterations = 4;

	struct Edge {
		uint32_t i;
		uint32_t j;
		float    rest;
		float    lambda;
	};
	static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");
	std::vector<Edge> edges;
	std::array<uint32_t, 6> pass_offsets;  // cloth 4 + softbody 1
	std::vector<std::pair<uint32_t, uint32_t>> passes[5];

	// SoftBody Stretch Edge
	std::vector<Edge> softbody_stretch_edges;

	struct Shear {
		uint32_t i0, i1, i2;
		float rest_dot;

		float lambda;
		float p0;
		float p1;
		float p2;
	};
	static_assert(sizeof(Shear) == 32, "Shear must be 32 bytes");
	std::vector<Shear> shears;

	struct Bend {
		uint32_t i0, i1, i2, i3;
		float rest_angle;
		float lambda;
		float p0;
		float p1;
	};
	static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");
	std::vector<Bend> bends;

	struct Area {
		uint32_t i0, i1, i2;
		float rest_area;
		glm::vec3 rest_normal;
		float lambda;
	};
	static_assert(sizeof(Area) == 32, "Area must be 32 bytes");
	std::vector<Area> areas;

	struct EdgeKey
	{
		uint32_t a, b; // always a < b

		bool operator==(const EdgeKey& o) const noexcept {
			return a == o.a && b == o.b;
		}
	};

	struct EdgeKeyHash
	{
		size_t operator()(const EdgeKey& k) const noexcept {
			return (size_t(k.a) << 32) ^ size_t(k.b);
		}
	};

	struct TriRef
	{
		uint32_t triIndex;
		uint32_t oppVertex;
	};

	std::vector<Collider> colliders;

	float ComputeRestBendAngle(
		uint32_t i1, uint32_t i2,
		uint32_t i3, uint32_t i4,
		const std::vector<glm::vec4>& pos)
	{
		glm::vec3 p1 = glm::vec3(pos[i1]);
		glm::vec3 p2 = glm::vec3(pos[i2]);
		glm::vec3 p3 = glm::vec3(pos[i3]);
		glm::vec3 p4 = glm::vec3(pos[i4]);

		glm::vec3 e = p2 - p1;
		float el = glm::length(e);
		if (el < 1e-8f) return 0.0f;
		glm::vec3 ehat = e / el;

		glm::vec3 n1 = glm::normalize(glm::cross(p2 - p1, p3 - p1));
		glm::vec3 n2 = glm::normalize(glm::cross(p2 - p1, p4 - p1));

		float c = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
		glm::vec3 cross_n1n2 = glm::cross(n1, n2);
		float s = glm::dot(ehat, cross_n1n2);

		float phi = std::atan2(s, c); // atan(s,c)¿Í µ¿ÀÏ

		return phi;
	}

	void BuildBendConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices)
	{
		bends.clear();

		std::unordered_map<EdgeKey, std::pair<TriRef, TriRef>, EdgeKeyHash> edgeMap;
		edgeMap.reserve(indices.size());

		auto make_edge = [](uint32_t i, uint32_t j) {
			EdgeKey k;
			if (i < j) { k.a = i; k.b = j; }
			else { k.a = j; k.b = i; }
			return k;
			};

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			uint32_t i0 = indices[i];
			uint32_t i1 = indices[i + 1];
			uint32_t i2 = indices[i + 2];

			// tri edges: (i0,i1), (i1,i2), (i2,i0)
			EdgeKey e01 = make_edge(i0, i1);
			EdgeKey e12 = make_edge(i1, i2);
			EdgeKey e20 = make_edge(i2, i0);

			TriRef r0{ i, i2 };
			TriRef r1{ i, i0 };
			TriRef r2{ i, i1 };

			auto insert_ref = [&](const EdgeKey& e, const TriRef& r)
				{
					auto it = edgeMap.find(e);
					if (it == edgeMap.end())
					{
						edgeMap.emplace(e, std::make_pair(r, TriRef{ UINT32_MAX, UINT32_MAX }));
					}
					else
					{
						if (it->second.second.triIndex == UINT32_MAX)
							it->second.second = r;
					}
				};

			insert_ref(e01, r0);
			insert_ref(e12, r1);
			insert_ref(e20, r2);
		}

		bends.reserve(edgeMap.size());

		for (const auto& kv : edgeMap)
		{
			const EdgeKey& e = kv.first;
			const TriRef& t0 = kv.second.first;
			const TriRef& t1 = kv.second.second;

			if (t1.triIndex == UINT32_MAX)
				continue;

			uint32_t i1 = e.a;         // hinge start
			uint32_t i2 = e.b;         // hinge end
			uint32_t i3 = t0.oppVertex;// tri0 opposite
			uint32_t i4 = t1.oppVertex;// tri1 opposite

			float restAngle = ComputeRestBendAngle(i1, i2, i3, i4, positions);

			Bend bc;
			bc.i0 = i1;
			bc.i1 = i2;
			bc.i2 = i3;
			bc.i3 = i4;
			bc.rest_angle = restAngle;
			bc.lambda = 0.0f;

			bends.push_back(bc);
		}

		num_bends = static_cast<uint32_t>(bends.size());
	}


	void BuildAreaConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices)
	{
		for (size_t i = 0; i < indices.size(); i += 3)
		{
			uint32_t i0 = indices[i];
			uint32_t i1 = indices[i + 1];
			uint32_t i2 = indices[i + 2];

			glm::vec3 p0 = positions[i0];
			glm::vec3 p1 = positions[i1];
			glm::vec3 p2 = positions[i2];

			glm::vec3 e0 = p1 - p0;
			glm::vec3 e1 = p2 - p0;

			glm::vec3 restNormal = glm::cross(e0, e1);
			float restArea = 0.5f * glm::length(restNormal);

			glm::vec3 nHat = (restArea > 0.0f) ? (restNormal / (2.0f * restArea)) : glm::vec3(0, 1, 0);

			Area area;
			area.i0 = i0;
			area.i1 = i1;
			area.i2 = i2;
			area.lambda = 0.0f;
			area.rest_area = restArea;
			area.rest_normal = nHat;

			areas.push_back(area);
		}

		num_areas = static_cast<uint32_t>(areas.size());
	}

	void ResetConstraints(std::vector<glm::vec4>& positions, std::vector<uint32_t>& indices)
	{
		// Edge - Stretch, Diagonal
		{
			uint32_t idx = 0;
			for (int p = 0; p < pass_offsets.size(); ++p) {
				for (auto [i, j] : passes[p]) {
					glm::vec3 pi = glm::vec3(positions[i]);
					glm::vec3 pj = glm::vec3(positions[j]);
					float rest = glm::length(pj - pi);
					edges[idx].rest = rest;
					edges[idx].lambda = 0.0f;
					idx++;
				}
			}
		}

		// Shear
		{
			for (int i = 0; i < num_shears; i++)
			{
				shears[i].lambda = 0.0f;
			}

		}

		// Bend
		{
			for (int i = 0; i < num_bends; i++)
			{
				bends[i].lambda = 0.0f;
			}
		}

	}

	struct SSBO {
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer delta_count{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer grab_state{ nullptr };
		vk::raii::Buffer area{ nullptr };
		vk::raii::Buffer volume{ nullptr };
		vk::raii::Buffer collider{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory delta_count{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory grab_state{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };
		vk::raii::DeviceMemory volume{ nullptr };
		vk::raii::DeviceMemory collider{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		vk::DeviceSize delta_x = 0;
		vk::DeviceSize delta_y = 0;
		vk::DeviceSize delta_z = 0;
		vk::DeviceSize delta_count = 0;
		vk::DeviceSize edge = 0;
		vk::DeviceSize shear = 0;
		vk::DeviceSize bend = 0;
		vk::DeviceSize grab_state = 0;
		vk::DeviceSize area = 0;
		vk::DeviceSize volume = 0;
		vk::DeviceSize collider = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer area{ nullptr };
		vk::raii::Buffer collider{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory area{ nullptr };
		vk::raii::DeviceMemory collider{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* edge{ nullptr };
		void* shear{ nullptr };
		void* bend{ nullptr };
		void* area{ nullptr };
		void* collider{ nullptr };
	} staging_mapped_;
};