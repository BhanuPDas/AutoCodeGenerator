package de.fhdortmund.codegenerator;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Scope;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import redis.clients.jedis.UnifiedJedis;

@SpringBootApplication
public class CodeGeneratorApplication {

    @Value("${redis.url}")
    private String redisUrl;

    public static void main(String[] args) {
        SpringApplication.run(CodeGeneratorApplication.class, args);
    }

    @Bean
    @Scope(value = "singleton")
    public RestTemplate createRestTemplate(RestTemplateBuilder restBuilder) {
        return restBuilder.requestFactory(() -> new HttpComponentsClientHttpRequestFactory(
                HttpClients.custom().setConnectionReuseStrategy((request, response, context) -> false).build())).build();
    }

    @Bean
    @Scope(value = "singleton")
    public UnifiedJedis stringRedisTemplate() {
        return new UnifiedJedis(redisUrl);
    }

}
