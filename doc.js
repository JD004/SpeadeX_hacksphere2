import React, { useState } from 'react';
import { Send, Mic, Menu } from 'lucide-react';

const ChatDocClone = () => {
  const [messages, setMessages] = useState([
    { text: 'Welcome to ChatDoc! How can I assist you today?', sender: 'bot' },
    { text: "Hi, I'm looking for some help.", sender: 'user' },
  ]);
  const [input, setInput] = useState('');
  const [micActive, setMicActive] = useState(false);

  const sendMessage = () => {
    if (input.trim() === '') return;
    setMessages([...messages, { text: input, sender: 'user' }]);
    setInput('');
  };

  const activateMic = () => {
    setMicActive(true);
    setTimeout(() => setMicActive(false), 3000);
  };

  return (
    <div className="flex h-screen bg-black text-white">
      {/* Sidebar */}
      <div className="w-72 bg-black p-6 hidden md:flex flex-col">
        <div className="flex items-center mb-8">
          <div className="w-10 h-10 rounded-full bg-blue-900 flex items-center justify-center">
            <span className="text-lg font-bold">CD</span>
          </div>
          <h2 className="ml-3 text-2xl font-semibold">ChatDoc</h2>
        </div>
        <nav className="flex-1">
          <ul>
            <li className="mb-4 hover:bg-blue-800 p-2 rounded cursor-pointer">Chats</li>
            <li className="mb-4 hover:bg-blue-800 p-2 rounded cursor-pointer">Contacts</li>
            <li className="mb-4 hover:bg-blue-800 p-2 rounded cursor-pointer">Settings</li>
          </ul>
        </nav>
        <div className="mt-auto">
          <p className="text-sm text-gray-500">© 2025 ChatDoc</p>
        </div>
      </div>

      {/* Main Chat Section */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 flex items-center justify-between border-b border-blue-900">
          <button className="md:hidden">
            <Menu size={24} className="text-white" />
          </button>
          <div className="flex items-center">
            <img
              src="https://via.placeholder.com/40"
              alt="User Avatar"
              className="w-10 h-10 rounded-full mr-3"
            />
            <div>
              <p className="text-lg font-semibold">User Name</p>
              <p className="text-xs text-blue-400">Online</p>
            </div>
          </div>
        </div>

        {/* Mic Button (Centered) */}
        <div className="py-4 flex justify-center">
          <button
            onClick={activateMic}
            className={`bg-blue-900 p-4 rounded-full transition duration-200 ${
              micActive ? 'ring-4 ring-blue-500 animate-pulse' : 'hover:bg-blue-800'
            } ripple`}
          >
            <Mic size={32} className="text-white" />
          </button>
        </div>

        {/* Rotating Row of 8 Images */}
        <div className="overflow-hidden relative py-4 w-full">
          <div className="flex space-x-4 animate-marquee">
            {[...Array(8)].map((_, i) => (
              <img
                key={i}
                src={`https://via.placeholder.com/80/1E3A8A/FFFFFF?text=Img${i + 1}`}
                alt={`Img ${i + 1}`}
                className="w-20 h-20"
              />
            ))}
          </div>
        </div>

        {/* Chatbox / Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`p-3 rounded-lg max-w-xs shadow-md transition-all duration-200 ${
                  msg.sender === 'user'
                    ? 'bg-blue-900 hover:bg-blue-800'
                    : 'bg-blue-950 hover:bg-blue-900'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-blue-900 flex items-center space-x-3">
          <input
            type="text"
            className="flex-1 p-3 bg-blue-950 border border-blue-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-700"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button
            onClick={sendMessage}
            className="bg-blue-800 p-3 rounded-full hover:bg-blue-700 transition duration-200"
          >
            <Send size={20} className="text-white" />
          </button>
        </div>
      </div>

      {/* CSS for Marquee Animation and Ripple Effect */}
      <style jsx>{`
        @keyframes marquee {
          from { transform: translateX(100%); }
          to { transform: translateX(-100%); }
        }
        .animate-marquee {
          display: flex;
          animation: marquee 10s linear infinite;
        }
        .ripple {
          position: relative;
          overflow: hidden;
        }
        .ripple::after {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 100%;
          height: 100%;
          background: rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          transform: scale(0);
          transition: transform 0.5s, opacity 1s;
          pointer-events: none;
        }
        .ripple:active::after {
          transform: scale(2);
          opacity: 0;
          transition: transform 0.5s, opacity 0s;
        }
      `}</style>
    </div>
  );
};

export default ChatDocClone;
