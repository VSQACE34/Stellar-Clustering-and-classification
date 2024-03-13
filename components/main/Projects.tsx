import React from "react";
import ProjectCard from "../sub/ProjectCard";

const Projects = () => {
  return (
    <div
      className="flex flex-col items-center justify-center py-20"
      id="projects"
    >
      <h1 className="text-[40px] font-semibold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-cyan-500 py-20">
        Other Projects
      </h1>
      <div className="h-auto w-auto flex flex-col md:flex-row gap-10 px-10">
        <ProjectCard
          src="/CardImage.png"
          title="Project 1"
          description="description of the project."/>
        <ProjectCard
          src="/full_screenshot.png"
          title="Project 2"
          description="description of the project."/>
        <ProjectCard
          src="/SpaceWebsite(1).png"
          title="Project 3"
          description="description of the project."/>
      </div>
    </div>
  );
};

export default Projects;
