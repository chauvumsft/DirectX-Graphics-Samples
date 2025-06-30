#pragma once
#include "RaytracingHlslCompat.h"

class ObjLoader;

class ObjectConstantBuffer
{
	uint32_t materialID;
	XMFLOAT4 albedo;

public:
	void LoadObjMesh(
		std::string name,
		float scale,
		ObjLoader* loader,
		DX::DeviceResources* deviceResources,
		UINT descriptorSize,
		XMMATRIX transform,
		std::vector<Vertex>* floorVertices,
		std::vector<Index>* indices);
};