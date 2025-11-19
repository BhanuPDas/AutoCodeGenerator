package de.fhdortmund.codegenerator.requests;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.io.Serializable;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class Prompts implements Serializable {
    @NotNull(message = "Null Prompts")
    @NotBlank(message = "Empty Prompts")
    private String prompt;
}
