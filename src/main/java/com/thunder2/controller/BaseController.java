package com.thunder2.controller;

import com.thunder2.dto.TorchInferenceDto;
import com.thunder2.service.impl.TorchInferenceServiceImpl;
import com.thunder2.vo.RestMessage;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/torch")
public class BaseController {

    @Autowired
    TorchInferenceServiceImpl torchInferenceService;


    @PostMapping("/program")
    public RestMessage<List<Object>> inference(@RequestBody TorchInferenceDto torchInferenceDto){
        List<Object> listObject = new ArrayList<>();
        listObject = torchInferenceService.inference(torchInferenceDto.getInput(), listObject);

        return RestMessage.newInstance(0, "OK", listObject);
        //        listObject = module.infer((int)2e6, input);
    }
}
