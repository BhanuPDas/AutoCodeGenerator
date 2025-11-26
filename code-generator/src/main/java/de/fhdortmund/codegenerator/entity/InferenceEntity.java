package de.fhdortmund.codegenerator.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "results", schema = "llm")
public class InferenceEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(name = "msg_id")
    private String msgId;
    @Column(name = "inf_req")
    private String prompt;
    @Column(name = "inf_resp")
    private String result;
    @Column(name = "send_to_ui")
    private String sendToUi;
}
