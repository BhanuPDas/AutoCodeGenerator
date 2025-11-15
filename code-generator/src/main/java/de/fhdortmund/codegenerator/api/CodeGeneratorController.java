package de.fhdortmund.codegenerator.api;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/")
public class CodeGeneratorController {

    @PostMapping("/query")
    public void query() {

    }
}
