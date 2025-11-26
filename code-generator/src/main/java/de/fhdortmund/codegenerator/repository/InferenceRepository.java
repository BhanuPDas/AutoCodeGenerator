package de.fhdortmund.codegenerator.repository;

import de.fhdortmund.codegenerator.entity.InferenceEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface InferenceRepository extends JpaRepository<InferenceEntity, Long> {
    List<InferenceEntity> findBySentToUiEqualsIgnoreCase(String sentToUi);
}
