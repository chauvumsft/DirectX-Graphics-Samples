#pragma once
#include "RaytracingHlslCompat.h"
#include "CheckCast.h"

class ObjLoader;

class ObjectConstantBuffer
{

	uint32_t materialID;
	XMFLOAT4 albedo;

public:
	void Initialize(XMFLOAT4 albedo, uint32_t materialID);

	void LoadCube(
		ObjLoader* loader,
		float xScale,
		float yScale,
		float zScale,
		float xTranslate,
		float yTranslate,
		float zTranslate,
		float uvScale,
		DX::DeviceResources* deviceResources,
		UINT descriptorSize,
		std::vector<Vertex>* floorVertices,
		std::vector<Index>* indices);

	void LoadObjMesh(
		std::string name,
		float scale,
		ObjLoader* loader,
		DX::DeviceResources* deviceResources,
		UINT descriptorSize,
		XMMATRIX transform,
		std::vector<Vertex>* floorVertices,
		std::vector<Index>* indices);

	//D3D12_RAYTRACING_GEOMETRY_DESC GetRaytracingGeometryDesc(D3DBuffer* vertexBuffer, D3DBuffer* indexBuffer, int totalVertexCount);



	//uint32_t GetIndexBufferOffset();

	//uint32_t GetMaterial();



};