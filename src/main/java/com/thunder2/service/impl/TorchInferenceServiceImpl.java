package com.thunder2.service.impl;
import com.thunder2.service.TorchInferenceService;
import com.thunder2.utils.TmsrModule;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.List;

@Service
public class TorchInferenceServiceImpl implements TorchInferenceService{
    private static TmsrModule module = new TmsrModule("/home/jhy/repos/TorchDemo/src/main/java/com/thunder2/utils/scripted_icptxlp_e200_4cls.pt");

    @Override
    public List<Object> inference(float []input, List<Object> listObject){
        System.out.println(input);
        listObject = module.infer((int)2e6, input);
        return listObject;
    }
}
