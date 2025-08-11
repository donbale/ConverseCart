// Note: The 'import React...' line has been removed.
// When using React via CDN scripts in index.html, the React object and its hooks 
// (useState, useContext, createContext) are automatically available in the global scope.
const { useState, useContext, createContext, useEffect, useRef } = React;

// --- MOCK DATA ---
const productData = [
  {
    id: 'prod_001',
    name: 'Quantum Laptop',
    price: 1299.99,
    description: 'A sleek, powerful laptop with a next-generation quantum processor. Perfect for developers and creators.',
    category: 'Electronics',
    stock: 15,
    imageUrl: 'https://placehold.co/600x400/93c5fd/333?text=Quantum+Laptop'
  },
  {
    id: 'prod_002',
    name: 'Ergo-Mechanical Keyboard',
    price: 159.50,
    description: 'A split ergonomic keyboard with satisfying mechanical switches. Type for hours in comfort.',
    category: 'Peripherals',
    stock: 45,
    imageUrl: 'https://placehold.co/600x400/a5b4fc/333?text=Ergo+Keyboard'
  },
  {
    id: 'prod_003',
    name: '4K Ultra-Wide Monitor',
    price: 799.00,
    description: 'Immerse yourself in stunning detail with this 34-inch ultra-wide 4K monitor.',
    category: 'Monitors',
    stock: 22,
    imageUrl: 'https://placehold.co/600x400/818cf8/333?text=4K+Monitor'
  },
  {
    id: 'prod_004',
    name: 'AI-Powered Smart Mug',
    price: 89.99,
    description: 'Keeps your drink at the perfect temperature, controlled by an app. The future of beverage consumption.',
    category: 'Gadgets',
    stock: 120,
    imageUrl: 'https://placehold.co/600x400/c4b5fd/333?text=Smart+Mug'
  },
  {
    id: 'prod_005',
    name: 'Noise-Cancelling Headphones',
    price: 349.99,
    description: 'Zone in on your work or music with industry-leading noise cancellation and premium sound.',
    category: 'Audio',
    stock: 50,
    imageUrl: 'https://placehold.co/600x400/e9d5ff/333?text=Headphones'
  },
  {
    id: 'prod_006',
    name: 'Portable SSD 1TB',
    price: 129.99,
    description: 'Blazing fast, pocket-sized storage for your files, games, and projects.',
    category: 'Storage',
    stock: 75,
    imageUrl: 'https://placehold.co/600x400/d8b4fe/333?text=Portable+SSD'
  }
];

// --- CART CONTEXT ---
const CartContext = createContext();

const CartProvider = ({ children }) => {
  const [cart, setCart] = useState([]);

  const addToCart = (product) => {
    setCart(prevCart => {
      const existingItem = prevCart.find(item => item.id === product.id);
      if (existingItem) {
        return prevCart.map(item =>
          item.id === product.id ? { ...item, quantity: item.quantity + 1 } : item
        );
      }
      return [...prevCart, { ...product, quantity: 1 }];
    });
  };

  const removeFromCart = (productId) => {
    setCart(prevCart => prevCart.filter(item => item.id !== productId));
  };

  return (
    <CartContext.Provider value={{ cart, addToCart, removeFromCart }}>
      {children}
    </CartContext.Provider>
  );
};

// --- COMPONENTS ---

function Header({ setPage, cartItemCount }) {
  // This component remains the same
  return (
    <header className="bg-white shadow-md sticky top-0 z-10">
      <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
        <div 
          className="text-2xl font-bold text-gray-800 cursor-pointer"
          onClick={() => setPage({ name: 'productList' })}
        >
          <span className="text-indigo-600">AI</span>-Store
        </div>
        <button 
          onClick={() => setPage({ name: 'cart' })} 
          className="relative flex items-center bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          <span className="ml-2">Cart</span>
          {cartItemCount > 0 && (
            <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
              {cartItemCount}
            </span>
          )}
        </button>
      </nav>
    </header>
  );
}

function ProductList({ setPage }) {
  // This component remains the same
  const { addToCart } = useContext(CartContext);
  return (
    <div className="container mx-auto px-6 py-8">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Our Products</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {productData.map(product => (
          <div key={product.id} className="bg-white rounded-lg shadow-lg overflow-hidden flex flex-col">
            <img src={product.imageUrl} alt={product.name} className="w-full h-56 object-cover"/>
            <div className="p-6 flex-grow flex flex-col">
              <h3 className="text-xl font-semibold text-gray-800 mb-2">{product.name}</h3>
              <p className="text-gray-600 flex-grow">{product.description}</p>
              <div className="mt-6 flex justify-between items-center">
                <span className="text-2xl font-bold text-indigo-600">${product.price.toFixed(2)}</span>
                <button onClick={() => setPage({ name: 'productDetail', productId: product.id })} className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300">View</button>
              </div>
              <button onClick={() => addToCart(product)} className="w-full mt-4 bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700">Add to Cart</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProductDetail({ productId, setPage }) {
  // This component remains the same
  const { addToCart } = useContext(CartContext);
  const product = productData.find(p => p.id === productId);
  if (!product) return <div>Product not found</div>;
  return (
    <div className="container mx-auto px-6 py-8">
      <button onClick={() => setPage({ name: 'productList' })} className="mb-6 text-indigo-600 hover:underline">&larr; Back to Products</button>
      <div className="bg-white rounded-lg shadow-lg overflow-hidden md:flex">
        <img src={product.imageUrl} alt={product.name} className="md:w-1/2 w-full h-auto object-cover"/>
        <div className="p-8 md:w-1/2 flex flex-col justify-center">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">{product.name}</h2>
          <p className="text-gray-600 text-lg mb-6">{product.description}</p>
          <div className="flex items-center justify-between">
            <span className="text-4xl font-bold text-indigo-600">${product.price.toFixed(2)}</span>
            <button onClick={() => addToCart(product)} className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700">Add to Cart</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Cart({ setPage }) {
  // This component remains the same
  const { cart, removeFromCart } = useContext(CartContext);
  const total = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  return (
    <div className="container mx-auto px-6 py-8">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Your Shopping Cart</h2>
      {cart.length === 0 ? (
        <div className="text-center py-10 bg-white rounded-lg shadow-md">
          <p className="text-gray-600 text-xl">Your cart is empty.</p>
          <button onClick={() => setPage({ name: 'productList' })} className="mt-6 bg-indigo-600 text-white px-6 py-3 rounded-lg">Start Shopping</button>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-md p-6">
          {cart.map(item => (
            <div key={item.id} className="flex items-center justify-between border-b py-4">
              <div className="flex items-center">
                <img src={item.imageUrl} alt={item.name} className="w-20 h-20 object-cover rounded-md mr-4"/>
                <div>
                  <h3 className="text-lg font-semibold">{item.name}</h3>
                  <p className="text-gray-600">${item.price.toFixed(2)} x {item.quantity}</p>
                </div>
              </div>
              <div className="flex items-center">
                <p className="text-lg font-bold mr-6">${(item.price * item.quantity).toFixed(2)}</p>
                <button onClick={() => removeFromCart(item.id)} className="text-red-500 hover:text-red-700">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
              </div>
            </div>
          ))}
          <div className="mt-6 text-right">
            <p className="text-2xl font-bold">Total: <span className="text-indigo-600">${total.toFixed(2)}</span></p>
            <button className="mt-4 bg-green-500 text-white px-8 py-3 rounded-lg">Proceed to Checkout</button>
          </div>
        </div>
      )}
    </div>
  );
}

// --- NEW CHAT WIDGET COMPONENT ---
function ChatWidget({ onAction }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([{ sender: 'bot', text: 'Hello! How can I help you shop today?' }]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage = { sender: 'user', text: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send the user's query to the FastAPI server
      const response = await fetch('http://localhost:8001/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputValue }),
      });
      const action = await response.json();
      
      // Execute the action returned by the server
      onAction(action);
      
      // Add a confirmation message from the bot
      const botMessage = { sender: 'bot', text: `Okay, I've handled that for you.` };
      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error("Error communicating with the RAG API:", error);
      const errorMessage = { sender: 'bot', text: "Sorry, I'm having trouble connecting. Please try again later." };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-5 right-5 z-20">
      {isOpen && (
        <div className="bg-white w-80 h-96 rounded-lg shadow-2xl flex flex-col">
          <div className="bg-indigo-600 text-white p-3 rounded-t-lg">
            <h3 className="font-semibold text-center">AI Shopping Assistant</h3>
          </div>
          <div className="flex-grow p-4 overflow-y-auto">
            {messages.map((msg, index) => (
              <div key={index} className={`flex mb-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`rounded-lg px-3 py-2 ${msg.sender === 'user' ? 'bg-indigo-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
                  {msg.text}
                </div>
              </div>
            ))}
            {isLoading && <div className="text-center text-gray-500">...</div>}
            <div ref={messagesEndRef} />
          </div>
          <form onSubmit={handleSendMessage} className="p-3 border-t">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask me to find or add items..."
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled={isLoading}
            />
          </form>
        </div>
      )}
      <button onClick={() => setIsOpen(!isOpen)} className="bg-indigo-600 text-white rounded-full p-4 shadow-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
      </button>
    </div>
  );
}


// --- APP STRUCTURE ---
function EcommerceApp() {
  const [page, setPage] = useState({ name: 'productList' });
  const { cart, addToCart } = useContext(CartContext);
  const cartItemCount = cart.reduce((sum, item) => sum + item.quantity, 0);

  // This function executes the command from the RAG pipeline
  const executeAction = (action) => {
    console.log("Executing action:", action);
    if (!action || !action.action || !action.payload) {
      console.warn("Received invalid action:", action);
      return;
    }

    switch (action.action) {
      case 'setPage':
        setPage(action.payload);
        break;
      case 'addToCart':
        // The RAG pipeline gives us the product ID. We find the full product object here.
        const productToAdd = productData.find(p => p.id === action.payload.id);
        if (productToAdd) {
          addToCart(productToAdd);
        } else {
          console.error("Could not find product with ID:", action.payload.id);
        }
        break;
      // You could add 'removeFromCart' here as well
      default:
        console.warn("Unknown action received:", action.action);
    }
  };

  const renderPage = () => {
    switch (page.name) {
      case 'productDetail':
        return <ProductDetail productId={page.productId} setPage={setPage} />;
      case 'cart':
        return <Cart setPage={setPage} />;
      case 'productList':
      default:
        return <ProductList setPage={setPage} />;
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen font-sans">
      <Header setPage={setPage} cartItemCount={cartItemCount} />
      <main>
        {renderPage()}
      </main>
      <ChatWidget onAction={executeAction} />
    </div>
  );
}

// The root component that provides the Cart context to the whole app.
function App() {
  return (
    <CartProvider>
      <EcommerceApp />
    </CartProvider>
  );
}
