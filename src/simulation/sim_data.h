#pragma once

struct SimData {
	const uint32_t nx = 60;
	const uint32_t ny = 60;

	glm::vec2 cloth_size{ 4.0f, 4.0f };
	float spacing_x = cloth_size.x / nx;
	float spacing_y = cloth_size.y / ny;
	float cloth_height = 6.0f;
	float mass = cloth_size.x * cloth_size.y * 0.5;

	struct Compliance {
		float stretch = 1e-9f;
		float shear = 1e-9f;
		float bend = 0.8f;
		float area = 0.8f;
		float self_collision = 1e-9f;
	} compliance;

	struct Beta {
		float stretch = 300.0f;
	} beta;

	uint32_t num_particles = nx * ny;
	uint32_t num_indices = 0;
	uint32_t num_edges = 0;
	uint32_t num_shears = 0;
	uint32_t num_bends = 0;
	uint32_t num_areas = 0;

	float frame_dt = 60.0f;
	int substeps = 10;
	int iterations = 4;

	std::vector<glm::vec4> positions;
	std::vector<glm::vec4> pred_positions;
	std::vector<glm::vec4> velocities;
	std::vector<float> inverse_masses;
	std::vector<float> masses;
	std::vector<uint32_t> indices;

	std::vector<uint32_t> particle_hashes;
	std::vector<uint32_t> particle_indices;
	std::vector<uint32_t> starts;
	std::vector<uint32_t> ends;

	uint32_t num_neighbors;
	std::vector<uint32_t> neighbors;
	std::vector<uint32_t> neighbor_lambdas;

	struct Edge {
		uint32_t i;
		uint32_t j;
		float    rest;
		float    lambda;
	};
	static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");
	std::vector<Edge> edges;
	std::array<uint32_t, 5> pass_offsets;
	std::vector<std::pair<uint32_t, uint32_t>> passes[5];

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

	void BuildBendConstraints()
	{
		bends.clear();

		const size_t numTris = indices.size() / 3;

		std::unordered_map<EdgeKey, std::pair<TriRef, TriRef>, EdgeKeyHash> edgeMap;
		edgeMap.reserve(numTris * 3);

		auto make_edge = [](uint32_t i, uint32_t j) {
			EdgeKey k;
			if (i < j) { k.a = i; k.b = j; }
			else { k.a = j; k.b = i; }
			return k;
			};

		for (uint32_t t = 0; t < numTris; ++t)
		{
			uint32_t i0 = indices[3 * t + 0];
			uint32_t i1 = indices[3 * t + 1];
			uint32_t i2 = indices[3 * t + 2];

			// tri edges: (i0,i1), (i1,i2), (i2,i0)
			EdgeKey e01 = make_edge(i0, i1);
			EdgeKey e12 = make_edge(i1, i2);
			EdgeKey e20 = make_edge(i2, i0);

			TriRef r0{ t, i2 };
			TriRef r1{ t, i0 };
			TriRef r2{ t, i1 };

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


	void BuildAreaConstraints()
	{
		areas.reserve(num_indices / 3);
		for (size_t t = 0; t < num_indices; t += 3)
		{
			uint32_t i0 = indices[t];
			uint32_t i1 = indices[t + 1];
			uint32_t i2 = indices[t + 2];

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
	}

	void ResetConstraints()
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

		// mass
		{
			// Total area
			float totalArea = 0.0f;
			for (size_t t = 0; t < num_indices; t += 3) {
				uint32_t i0 = indices[t + 0];
				uint32_t i1 = indices[t + 1];
				uint32_t i2 = indices[t + 2];

				glm::vec3 p0 = glm::vec3(positions[i0]);
				glm::vec3 p1 = glm::vec3(positions[i1]);
				glm::vec3 p2 = glm::vec3(positions[i2]);

				glm::vec3 e1 = p1 - p0;
				glm::vec3 e2 = p2 - p0;

				float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
				totalArea += area;
			}

			float totalMassTarget = mass;

			// Area zero defence
			float density = 0.0f;
			if (totalArea > 0.0f) {
				density = totalMassTarget / totalArea; // kg/m©÷
			}

			// Distribute mass to each triangle in proportion to area
			for (size_t t = 0; t < num_indices; t += 3) {
				uint32_t i0 = indices[t + 0];
				uint32_t i1 = indices[t + 1];
				uint32_t i2 = indices[t + 2];

				glm::vec3 p0 = glm::vec3(positions[i0]);
				glm::vec3 p1 = glm::vec3(positions[i1]);
				glm::vec3 p2 = glm::vec3(positions[i2]);

				glm::vec3 e1 = p1 - p0;
				glm::vec3 e2 = p2 - p0;

				float area = 0.5f * glm::length(glm::cross(e1, e2));

				float triMass = density * area;

				float share = triMass / 3.0f;
				masses[i0] += share;
				masses[i1] += share;
				masses[i2] += share;
			}

			// Set inverse masses using mass
			for (uint32_t i = 0; i < num_particles; ++i) {
				float m = masses[i];
				if (m > 0.0f)
					inverse_masses[i] = 1.0f / m;
				else
					inverse_masses[i] = 0.0f;
			}
		}
	}

};