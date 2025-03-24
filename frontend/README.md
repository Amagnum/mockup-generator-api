# T-Shirt Mockup Generator Frontend

This is the frontend application for the T-Shirt Mockup Generator API. It provides a user-friendly interface to generate realistic t-shirt mockups with custom designs.

## Features

- Upload source images, masks, depth maps, and design images
- Adjust design placement and scaling
- Choose t-shirt colors with a color picker
- Configure advanced coloring parameters
- Preview and download generated mockups

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory:
   ```
   cd frontend
   ```
3. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   yarn install
   ```

### Running the Development Server

```
npm start
```
or
```
yarn start
```

The application will be available at http://localhost:3000.

### Building for Production

```
npm run build
```
or
```
yarn build
```

This will create an optimized production build in the `build` directory.

## Configuration

The application connects to the T-Shirt Mockup Generator API. By default, it connects to `http://localhost:8000`. You can change this by setting the `REACT_APP_API_URL` environment variable.

## Usage

1. Upload the required images (source image, mask, depth map, and design)
2. Adjust the design placement, scaling, and t-shirt color
3. Configure advanced settings if needed
4. Generate the mockup
5. Preview and download the result

## Technologies Used

- React
- Material-UI
- react-colorful
- react-dropzone
- Axios 