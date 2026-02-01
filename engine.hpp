#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <set>
#include <algorithm>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;

    Value(double v, std::vector<std::shared_ptr<Value>> children = {}) 
        : data(v), grad(0.0), _prev(children) {
        _backward = []() {};
    }

    static std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
        auto out = std::make_shared<Value>(a->data + b->data, std::vector<std::shared_ptr<Value>>{a, b});
        out->_backward = [a, b, out]() {
            a->grad += out->grad;
            b->grad += out->grad;
        };
        return out;
    }

    static std::shared_ptr<Value> mul(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
        auto out = std::make_shared<Value>(a->data * b->data, std::vector<std::shared_ptr<Value>>{a, b});
        out->_backward = [a, b, out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }

    // Função de Ativação ReLU
    static std::shared_ptr<Value> relu(std::shared_ptr<Value> a) {
        auto out = std::make_shared<Value>(a->data < 0 ? 0 : a->data, std::vector<std::shared_ptr<Value>>{a});
        out->_backward = [a, out]() {
            a->grad += (out->data > 0) ? out->grad : 0;
        };
        return out;
    }

    // O motor de retropropagação (Autograd)
    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>> visited;
        
        std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& prev : v->_prev) {
                    build_topo(prev);
                }
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());
        this->grad = 1.0;
        std::reverse(topo.begin(), topo.end());
        for (auto& v : topo) {
            v->_backward();
        }
    }
};
