import React, { useRef, useState, useCallback, useEffect } from 'react';

const BoundingBoxDrawer = ({ 
  imageFile, 
  onBoundingBoxChange, 
  initialBox = { x1: 0.2, y1: 0.2, x2: 0.8, y2: 0.8 }
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState(null);
  const [currentBox, setCurrentBox] = useState(initialBox);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Load image when imageFile changes
  useEffect(() => {
    if (imageFile && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const url = URL.createObjectURL(imageFile);
      const img = new Image();
      
      img.onload = () => {
        // Set canvas size to match container while maintaining aspect ratio
        const containerWidth = canvas.offsetWidth;
        const aspectRatio = img.naturalWidth / img.naturalHeight;
        const containerHeight = Math.min(384, containerWidth / aspectRatio); // max 384px height
        
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        setImageLoaded(true);
        drawCanvas(img, currentBox);
        
        // Store image reference
        imageRef.current = img;
      };
      
      img.src = url;

      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [imageFile]);

  // Redraw when currentBox changes
  useEffect(() => {
    if (imageLoaded && imageRef.current) {
      drawCanvas(imageRef.current, currentBox);
    }
  }, [currentBox, imageLoaded]);

  const drawCanvas = (img, box) => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image to fill canvas
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw bounding box
    const x1 = box.x1 * canvas.width;
    const y1 = box.y1 * canvas.height;
    const x2 = box.x2 * canvas.width;
    const y2 = box.y2 * canvas.height;

    // Draw bounding box rectangle
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw corner handles
    const handleSize = 10;
    ctx.fillStyle = '#ef4444';
    ctx.setLineDash([]);
    
    // Corner handles
    const handles = [
      [x1, y1], // top-left
      [x2, y1], // top-right
      [x1, y2], // bottom-left
      [x2, y2]  // bottom-right
    ];
    
    handles.forEach(([x, y]) => {
      ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
    });

    // Draw semi-transparent overlay outside bounding box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    
    // Top
    ctx.fillRect(0, 0, canvas.width, y1);
    // Bottom
    ctx.fillRect(0, y2, canvas.width, canvas.height - y2);
    // Left
    ctx.fillRect(0, y1, x1, y2 - y1);
    // Right
    ctx.fillRect(x2, y1, canvas.width - x2, y2 - y1);
  };

  const getCanvasCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const x = (clientX - rect.left) / rect.width;
    const y = (clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  };

  const handleMouseDown = (e) => {
    if (!imageLoaded) return;
    
    const coords = getCanvasCoordinates(e);
    setStartPoint(coords);
    setIsDrawing(true);
  };

  const handleTouchStart = (e) => {
    e.preventDefault();
    handleMouseDown(e);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !startPoint || !imageLoaded) return;

    const coords = getCanvasCoordinates(e);
    const newBox = {
      x1: Math.min(startPoint.x, coords.x),
      y1: Math.min(startPoint.y, coords.y),
      x2: Math.max(startPoint.x, coords.x),
      y2: Math.max(startPoint.y, coords.y)
    };

    setCurrentBox(newBox);
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    handleMouseMove(e);
  };

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
      setStartPoint(null);
      onBoundingBoxChange(currentBox);
    }
  };

  const handleTouchEnd = (e) => {
    e.preventDefault();
    handleMouseUp();
  };

  const resetBox = () => {
    const defaultBox = { x1: 0.2, y1: 0.2, x2: 0.8, y2: 0.8 };
    setCurrentBox(defaultBox);
    onBoundingBoxChange(defaultBox);
  };

  if (!imageFile) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center min-h-96 flex items-center justify-center">
        <p className="text-gray-500">Upload an image to define bounding box</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-gray-100 rounded-lg p-4">
        <canvas
          ref={canvasRef}
          className="w-full h-64 sm:h-80 md:h-96 border border-gray-300 rounded cursor-crosshair bg-white touch-none"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>
      
      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-600">
          <p>Click and drag to define segmentation region</p>
          <p className="text-xs">Box: ({(currentBox.x1 * 100).toFixed(1)}%, {(currentBox.y1 * 100).toFixed(1)}%) → ({(currentBox.x2 * 100).toFixed(1)}%, {(currentBox.y2 * 100).toFixed(1)}%)</p>
        </div>
        <button
          onClick={resetBox}
          className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded transition-colors"
        >
          Reset Box
        </button>
      </div>
    </div>
  );
};

export default BoundingBoxDrawer;