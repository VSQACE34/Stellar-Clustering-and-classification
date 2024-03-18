"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  slideInFromLeft,
  slideInFromRight,
  slideInFromTop,
} from "@/utils/motion";
import { SparklesIcon } from "@heroicons/react/24/solid";
import Image from "next/image";

const HeroContent = () => {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      className="flex flex-row items-center justify-center px-20 mt-[-40px] w-full z-[10]"
    >
      <div className="h-full w-full flex flex-col gap-2 justify-center m-auto text-start">
        <motion.div
          variants={slideInFromTop}
          className="Welcome-box py-[8px] px-[7px] border border-[#7042f88b] opacity-[0.9]"
        >
          <SparklesIcon className="text-[#b49bff] mr-[5px] h-5 w-7" />
          <h1 className="Welcome-text text-[12px]">
            Project Overview
          </h1>
        </motion.div>

        <motion.div
          variants={slideInFromLeft(0.5)}
          className="flex flex-col gap-6 mt-6 text-4xl font-bold text-white max-w-[600px] w-auto h-auto"
        >
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-cyan-500">
            {" "}
              Stellar Classification and Clustering{" "}
          </span>
        </motion.div>

        <motion.p
          variants={slideInFromLeft(0.8)}
          className="text-lg text-gray-400 my-[-2] max-w-[600px]"
        >
         The project introduces a stellar classifier and clustering model that employs XGBoost 
         and K-Means clustering to categorize stellar objects into three main categories: stars,
         quasars (QSOs), and galaxies, based on u, g, r, i, z photometric bands and redshift 
         values sourced from the Sloan Digital Sky Survey (SDSS) database. This tool is designed
         to aid astronomers and astrophysicists in the identification and preliminary 
         classification of these stellar objects. Given the model&apos;s reliance on specific data 
         types, it is inherently limited in classifying objects that fall outside these parameters,
         such as planets or other celestial bodies not emitting in the observed bands or exhibiting
         different redshift characteristics. This limitation presents a valuable opportunity for 
         further research, potentially expanding the model&apos;s classification capabilities or exploring
         new methods to incorporate a wider range of stellar phenomena
        </motion.p>

      </div>
      <motion.div
        variants={slideInFromRight(0.8)}
        className="w-full h-full flex justify-center items-center"
      >
        <Image
          src="/nothing.svg"
          alt="work around"
          height={650}
          width={650}
        />
      </motion.div>
    </motion.div>
  );
};

export default HeroContent;
