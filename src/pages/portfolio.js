import React from "react"
import { Link } from "gatsby"

import Layout from "../components/layout"
import SEO from "../components/seo"

const Portfolio = () => (
  <Layout>
    <SEO title="narenandu portfolio" />
    <h1>Portfolio</h1>
    <p>
      <a href="https://sites.google.com/site/narenportfolio/">
        narenandu portfolio
      </a>
    </p>
    <p>Getting Embedded here soon !</p>
    <hr></hr>
    <Link to="/">Naren Home</Link>
  </Layout>
)

export default Portfolio
