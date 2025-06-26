#include "stdafx.h"
#include "RaytracingHlslCompat.h"
#include "CubeConstantBuffer.h"
#include "DirectXRaytracingHelper.h"
#include "D3D12RaytracingSakuraScene.h"
#include "ObjLoader.h"


void CubeConstantBuffer::Initialize(uint32_t material)
{
    materialID = material;
    albedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f); // default white
}

void CubeConstantBuffer::LoadCube(
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
    std::vector<Index>* indices)
{
    // TODO: Implement cube loading logic here
}

void CubeConstantBuffer::LoadObjMesh(
	std::string name,
	float scale,
	ObjLoader* loader,
	DX::DeviceResources* deviceResources,
	UINT descriptorSize,
	XMMATRIX transform,
	std::vector<Vertex>* vertices,
	std::vector<Index>* indices)
{
	size_t vertexBaseline = vertices->size();
	size_t indexBaseline = indices->size();
	loader->GetObjectVerticesAndIndices(name, scale, vertices, indices);

	//m_vertexCount = vertices->size() - vertexBaseline;
	//m_indexCount = indices->size() - indexBaseline;
	//m_vertexBufferOffset = vertexBaseline * sizeof(Vertex);
	//m_indexBufferOffset = indexBaseline * sizeof(Index);

	//assert(m_indexBufferOffset % 6 == 0);

	//m_baseTransform = transform;
	//CreateTransformBuffer(deviceResources, m_baseTransform);
}