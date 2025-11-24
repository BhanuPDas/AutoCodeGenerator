FROM maven:3.9.7-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:resolve
COPY src ./src
RUN mvn clean compile package -DskipTests=true

# Stage 2: Run the application
FROM eclipse-temurin:21-jdk-jammy
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
EXPOSE 4010
ENTRYPOINT ["java", "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED", "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED", "--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED", "--add-exports=jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED", "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED", "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED", "-jar", "app.jar"]
