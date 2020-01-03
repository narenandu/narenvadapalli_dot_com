/**
 * Layout component that queries for data
 * with Gatsby's useStaticQuery component
 *
 * See: https://www.gatsbyjs.org/docs/use-static-query/
 */

import React from "react"
import PropTypes from "prop-types"
import { useStaticQuery, graphql, Link} from "gatsby"

import Header from "./header"
import "./layout.css"

const Layout = ({ children }) => {
  const data = useStaticQuery(graphql`
    query SiteTitleQuery {
      site {
        siteMetadata {
          title
        }
      }
    }
  `)

  return (
    <>
      <div style={{position: `fixed`,
              zIndex:`-99`,
              width: `100%`,
              height: `100%`,
            }}
      >
        <iframe frameborder="0" height="100%" width="100%" 
          src="https://youtube.com/embed/Dxs_xPKdr6E?autoplay=1&controls=0&showinfo=0&autohide=1&cc_load_policy=1">
        </iframe>
      </div>          

      <Header siteTitle={data.site.siteMetadata.title} />
      <div
        style={{
          margin: `0 auto`,
          maxWidth: 960,
          padding: `0px 1.0875rem 1.45rem`,
          paddingTop: 0,
        }}
      >
        <main>{children}</main>
        <hr></hr>
        <Link to="/">Naren Home</Link>
        <footer>© {new Date().getFullYear()}, narenVadapalli</footer>
      </div>
  
    </>
  )
}

Layout.propTypes = {
  children: PropTypes.node.isRequired,
}

export default Layout
