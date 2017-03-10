#pragma once

class Layer {
public:
	int BatchSize;

	Layer() {
	}

	virtual void Forward() {}
	virtual void Backward() {}
	virtual void Allocate() {}
	virtual void Free() {}
	virtual void UpdateParameter() {}
	virtual void SetX(void*) {}
	virtual void SetY(void*) {}
	virtual void* GetY() { return NULL; }
};

