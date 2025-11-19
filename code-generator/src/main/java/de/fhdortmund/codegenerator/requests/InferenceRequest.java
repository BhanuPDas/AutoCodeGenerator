package de.fhdortmund.codegenerator.requests;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class InferenceRequest {

    private String prompt;
}
