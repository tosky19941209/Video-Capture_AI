"use client";
import Image from "next/image";
import { useState } from "react";
import api from '../service/axios'
export default function Home() {
  const [videoURL, setVideoUrl] = useState<string>();
  const [uploadVideo, setUploadVideo] = useState<any>(null)
  const SendVideo = () => {

    const formData = new FormData()
    formData.append('video', uploadVideo)

    // api.post("/facial_emotion")
    api.post('/facial_emotion', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then((res) => {
        console.log(res.data)
        const jsonData = JSON.stringify(res.data);
        const blob = new Blob([jsonData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'data.json';
        a.click();
      })
  }
  return (
    <div className="flex flex-col justify-center items-center w-[100vw] h-[100vh]">
      <input type="file" className=""
        onChange={(e) => {
          setVideoUrl(URL.createObjectURL(e.target.files[0]))
          setUploadVideo(e.target.files[0])
        }}
      />
      <video className="w-[40%] h-[40%] mt-[100px] bg-red-100" src={videoURL} controls></video>
      <button className="w-[400px] h-[50px] bg-red-300 mt-[100px]"
        onClick={SendVideo}>Send</button>
    </div>
  );
}
