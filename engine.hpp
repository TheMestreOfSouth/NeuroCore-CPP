#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

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

    // Operação de Soma
    static std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
        auto out = std::make_shared<Value>(a->data + b->data, std::vector<std::shared_ptr<Value>>{a, b});
        out->_backward = [a, b, out]() {
            a->grad += out->grad;
            b->grad += out->grad;
        };
        return out;
    }

    // Operação de Multiplicação
    static std::shared_ptr<Value> mul(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
        auto out = std::make_shared<Value>(a->data * b->data, std::vector<std::shared_ptr<Value>>{a, b});
        out->_backward = [a, b, out]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
        };
        return out;
    }
};
