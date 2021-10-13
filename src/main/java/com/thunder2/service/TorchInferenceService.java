package com.thunder2.service;

import java.util.List;

public interface TorchInferenceService {
    List<Object> inference(float []input, List<Object> listObject);
}
