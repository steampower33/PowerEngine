#pragma once

#include "mesh_data.h"

class GeometryGenerator {
public:

	static MeshData MakeSquare(const float scale = 1.0f);
	static MeshData MakeBox(const float scale = 1.0f);
	static MeshData MakeBoundsBox();
	static MeshData MakeCylinder(
		const float bottomRadius,
		const float topRadius, float height,
		int sliceCount);
	static MeshData MakeSphere(
		const float radius,
		const int numSlices, const int numStacks, const glm::vec2 texScale = { 1.0f, 1.0f });
	static void CalculateTangents(MeshData& meshData);

};