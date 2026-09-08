import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {title:"HUDYAT · Alphabet Practice",description:"Practice FSL alphabet poses with on-device recognition.",manifest:"/manifest.webmanifest",icons:{icon:"/icon-192.png",apple:"/icon-192.png"}};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="en"><body>{children}</body></html>;}
